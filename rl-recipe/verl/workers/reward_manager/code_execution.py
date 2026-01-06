# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import requests
from tqdm import tqdm
from functools import partial
from typing import Any, List, Dict, Tuple, Optional
from tqdm.asyncio import tqdm_asyncio
from transformers import PreTrainedTokenizer
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import ast
import re
import json
import random
import torch
import difflib
import fcntl  # 导入文件锁模块
import time

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.our_code.sandbox_eval.sandbox_utils import RunCodeRequest, SubmitRequest, TestConfig, submit_to_sandbox


# exec reward:
"""
Exec reward:
    no answer: -5
    compilation error: -2
    test case validation: 0-5
Format reward:
    -1, 1
"""
REWARD_CE = -2
REWARD_NON_ANSWER = -2


RUN_TIMEOUT = 10
MAX_REQUESTS = 64
LOC_MAX_REQUESTS = 256





class CodeExecutionRewardManager:
    """
    """

    def __init__(
        self,
        config,
        file,  # train_file or test_file
        tokenizer: PreTrainedTokenizer,
        num_examine=0,
        run_all_cases=True,
    ):
        self.config = config
        self.file = file
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.run_all_cases = run_all_cases

        if isinstance(self.file, str):
            self.dataset = pd.read_parquet(self.file)
        else:
            self.dataset = pd.concat([pd.read_parquet(f) for f in self.file])


        ### TODO: 现在的proxy_id是source_id，需要等全部数据去重后，加一个全局的ID
        self.dataset["proxy_id"] = self.dataset["id"]
        self.id_to_infos = self.dataset.set_index("proxy_id").to_dict(orient="index")

        # implement code hash so that we can reduce sandbox usage
        self.code_to_reward = {}
        self.executor = ThreadPoolExecutor(max_workers=32)  # 定义最大并发线程数
    
    def extract_question(self, prompt_str: str) -> Tuple[Optional[str], str]:
       
        # Split prompt_str to isolate question string
        if "<|im_start|>user" in prompt_str:
            system_prompt = prompt_str.split("<|im_start|>user", 1)[0]
            question_str = prompt_str.split("<|im_start|>user", 1)[1]
        elif "<|start_header_id|>user<|end_header_id|>" in prompt_str:
            system_prompt = prompt_str.split("<|start_header_id|>user<|end_header_id|>", 1)[0]
            question_str = prompt_str.split("<|start_header_id|>user<|end_header_id|>", 1)[1]
        else:
            question_str = prompt_str
        return question_str
    
    def desanitize(self, text: str) -> str:
        # Pattern to match code blocks starting with ```, with optional language identifier
        # and capturing everything after until the end or until another ```
        pattern = r"```(?:python)?\s*([\s\S]*?)(?:\s*```|$)"
        # Find all matches in the text
        matches = re.findall(pattern, text, re.IGNORECASE)

        # Return the first one
        return f"```python\n{matches[0]}\n```" if matches and len(matches[0]) > 0 else text
    
    def sanitize(self, text: str, extract_last: bool = False) -> str:
        # Pattern to find ```python ... ``` or ``` ... ``` and capture the content within.
        # The ([\s\S]*?) part is the content.
        pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"

        if extract_last:
            # Find all occurrences of the pattern. re.findall returns a list of the captured group's content.
            all_captured_contents = re.findall(pattern, text, re.IGNORECASE)
            if all_captured_contents:
                # Return the stripped content of the last found code block
                return all_captured_contents[-1].strip()
        else:
            # Find the first occurrence of the pattern.
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return the stripped content of the first code block
                return match.group(1).strip()
        
        # If no code block is found (or if extract_last is true and no blocks are found),
        # return the original text.
        return text
    
    def get_samples(self, ids: List[str]) -> List[pd.Series]:
        
        samples = []
        for task_id in ids:
            sample = self.id_to_infos[task_id]
            samples.append(sample)
        
        print("*" * 40)
        print(f"Len of samples:  {len(samples)}")
        print("*" * 40)
        
        for i in range(len(samples)):
            sample = samples[i]
            if i < self.num_examine:
                print("*" * 40)
                print("[TESTS: ]", sample["selected_uts"])
                print("*" * 40)
        
        return samples
        
    def check_ce(self, code_str: str) -> bool:
        
        if not isinstance(code_str, str):
            return True
        try:
            ast.parse(code_str)
            return False
        except:
            return True
        
    
    def parse_response(self, data: DataProto) -> DataProto:
        task_ids = []
        questions = []
        responses = []
        samples = []
        valid_response_lengths = []
        valid_response_idss = []

        for i in range(len(data)):
            data_item = data[i]
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            
            # # remove <eos> token
            # response_str = response_str.replace(self.tokenizer.eos_token, "")
            
            question = self.extract_question(prompt_str)
            task_id = data_item.non_tensor_batch["id"]

            
            task_ids.append(task_id)
            questions.append(question)
            responses.append(response_str)
            samples.append(data_item.non_tensor_batch)
            valid_response_lengths.append(valid_response_length)
            valid_response_idss.append(valid_response_ids)
            
            
        return task_ids, questions, responses, samples, valid_response_lengths, valid_response_idss
        
        
    def calculate_format_reward(self, processed_str: str, format_reward: int = 1):
        """Performs comprehensive validation of response structure.

        Args:
            processed_str: Processed response string from the model

        Returns:
            Boolean indicating whether all formatting requirements are met
        """
        
        debug_str = []
        debug_str.append("\n[Structure Validation]")
        validation_passed = True

        if self.config.actor_rollout_ref.model.distilled:
            return 0, debug_str




        # 检查唯一标签的位置
        unique_tags = {
            'answer': '<answer>',
            'answer_end': '</answer>'
        }

        positions = {}
        for tag_name, tag_str in unique_tags.items():
            count = processed_str.count(tag_str)
            positions[tag_name] = pos = processed_str.find(tag_str)
            
            debug_str.append(f"  {tag_str}: count={count}, position={pos}")
            
            if count != 1:
                debug_str.append(f"  [Error] {tag_str} appears {count} times (expected 1)")
                validation_passed = False

        # 验证基本顺序（think在开头，answer在结尾）
        if processed_str.strip()[0:len("<think>")] != "<think>":
            debug_str.append("  [Error] Incorrect start: Expected <think> at beginning")
            validation_passed = False
        elif positions['answer'] > positions['answer_end']:
            debug_str.append("  [Error] Incorrect tag order: Expected <answer>...</answer>")
            validation_passed = False
        elif not (processed_str.strip().endswith("</answer><|endoftext|>") or 
                processed_str.strip().endswith("</answer><|im_end|>") or
                processed_str.strip().endswith("</answer><|eot_id|>") or 
                processed_str.strip().endswith("</answer><|end_of_text|>")
                ):
            debug_str.append("  [Error] Incorrect ending: Expected </answer><|endoftext|> or </answer><|im_end|> or </answer><|eot_id|> or </answer><|end_of_text|>")
            validation_passed = False

        # 验证step和code的数量和顺序
        think_positions = [i for i in range(len(processed_str)) if processed_str.startswith('<think>', i)]
        think_end_positions = [i for i in range(len(processed_str)) if processed_str.startswith('</think>', i)]
        code_positions = [i for i in range(len(processed_str)) if processed_str.startswith('<code>', i)]
        code_end_positions = [i for i in range(len(processed_str)) if processed_str.startswith('</code>', i)]

        # 记录多次出现的标签信息
        debug_str.append(f"  <think> count: {len(think_positions)}")
        debug_str.append(f"  </think> count: {len(think_end_positions)}")
        debug_str.append(f"  <code> count: {len(code_positions)}")
        debug_str.append(f"  </code> count: {len(code_end_positions)}")

        # 验证每对标签的配对
        if len(think_end_positions) != len(think_positions):
            debug_str.append(f"  [Error] Number of </think> ({len(think_end_positions)}) does not match <think> ({len(think_positions)})")
            validation_passed = False

        if len(code_end_positions) != len(code_positions):
            debug_str.append(f"  [Error] Number of </code> ({len(code_end_positions)}) does not match <code> ({len(code_positions)})")
            validation_passed = False

        # 新增逻辑：检查 <answer> 和 </answer> 之间的内容是否包含 Python 代码块
        answer_start = processed_str.find('<answer>')
        answer_end = processed_str.find('</answer>')

        if answer_start != -1 and answer_end != -1:
            answer_content = processed_str[answer_start + len('<answer>'):answer_end]
            if "```python" not in answer_content and "```" not in answer_content:
                debug_str.append("  [Error] Python code block ' ```python ``` ' not found in <answer>...</answer>")
                validation_passed = False
            else:
                debug_str.append("  [Success] Python code block ' ```python ``` ' found in <answer>...</answer>")
        else:
            debug_str.append("  [Error] <answer> or </answer> not found, skipping Python code block check")
            validation_passed = False

        format_score = format_reward if validation_passed else -abs(format_reward)

        return format_score, debug_str
    


    ############################  oj reward   ############################
    def get_reward_all_oj(self, responses: list, samples: list, global_step: int = -1, batch_size: int = 16, max_parallel_threads: int = 16) -> list:
        exec_rewards = [REWARD_NON_ANSWER] * len(responses)
        total_batches = (len(responses) + batch_size - 1) // batch_size
        
        with ThreadPoolExecutor(max_workers=max_parallel_threads) as executor:
            future_to_index = {}

            for batch_num in range(total_batches):
                start_index = batch_num * batch_size
                end_index = min(start_index + batch_size, len(responses))
                responses_batch = responses[start_index:end_index]
                samples_batch = samples[start_index:end_index]
                future = executor.submit(self.process_batch, responses_batch, samples_batch)
                future_to_index[future] = (start_index, end_index)

            for future in as_completed(future_to_index):
                start_index, end_index = future_to_index[future]
                try:
                    batch_rewards = future.result()
                    exec_rewards[start_index:end_index] = batch_rewards
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
                    
        
        # format_reward calculation using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=512) as executor:
            format_rewards, debug_infos = zip(*list(executor.map(self.calculate_format_reward, responses)))
            format_rewards = list(format_rewards)
            debug_infos = list(debug_infos)
            
        # total_reward calculation
        total_rewards = [
            exec_reward + format_reward
            for exec_reward, format_reward in zip(exec_rewards, format_rewards)
        ]
                
        for i in range(len(samples)):
            if i < self.num_examine:
                print("*" * 40)
                # print("[SUBMISSIONS: ]", all_submissions)
                print("[GLOBAL_STEP: ]", global_step)
                print("[TASK_ID: ]", samples[i]["id"])
                print("[PROMPT: ]\n", samples[i]["problem"])
                print("[RESPONSE: ]\n", responses[i])
                print("-" * 20)
                print("\n".join(debug_infos[i]))
                print(f"  Format: {format_rewards[i]}")
                print(f"  Answer: {exec_rewards[i]}")
                print(f"  Total: {total_rewards[i]}")
                print("-" * 20)
                print("*" * 40)

        return total_rewards, exec_rewards, format_rewards            
                        
    def process_batch(self, responses_batch: list, samples_batch: list) -> list:
        batch_rewards = []
        all_submissions = []
        index_map = {}

        for index, (response, sample) in enumerate(zip(responses_batch, samples_batch)):
            
            if self.config.actor_rollout_ref.model.distilled:
                answer = response
                # For distilled model, extract the LAST code block.
                # sanitize will strip the content if a block is found.
                # If sanitize returns 'answer' (no block found), then .strip() applies to 'answer'.
                code_block = self.sanitize(answer, extract_last=True).strip()
            else:
                answer_pattern = r'<answer>(.*?)</answer>'
                matches = list(re.finditer(answer_pattern, response, re.DOTALL))
                if not matches:
                    batch_rewards.append(REWARD_NON_ANSWER)
                    # print("No Matcher REWARD_NON_ANSWER!!!")
                    continue
                answer = matches[-1].group(1).strip() # 'answer' is already stripped here
                # For non-distilled, extract the FIRST code block (default for sanitize).
                # sanitize will strip the content if a block is found.
                # If sanitize returns 'answer' (no block found), then .strip() applies to 'answer'.
                code_block = self.sanitize(answer).strip() 
                
            code_block = self.sanitize(answer).strip()
            if '\x00' in code_block:
                code_block = code_block.replace('\x00', '')

            if self.check_ce(code_block):
                batch_rewards.append(REWARD_CE)
                # print("REWARD_CE!!!")
                continue

            submissions = self.build_oj_submissions(code_block, sample)

            if not submissions:
                batch_rewards.append(REWARD_NON_ANSWER)
                print("No submissions REWARD_NON_ANSWER!!!")
                continue

            all_submissions.extend(submissions)
            index_map[index] = len(submissions)

        response_results = self.submit_batch(all_submissions)

        pos = 0
        for index, num_submissions in index_map.items():
            result_slice = response_results[pos:pos + num_submissions]
            pos += num_submissions
            batch_rewards.append(5 * np.mean(result_slice))

        return batch_rewards

    def build_oj_submissions(self, code_str: str, sample: dict) -> list:
        sample_size = 10
        submissions = []

        def build_submission(code_str, output_str=None, input_str=None) -> dict:
            submission = {
                "type": "python",
                "solution": code_str,
            }
            if output_str is not None:
                submission["expected_output"] = output_str
            if input_str is not None:
                submission["input"] = input_str
            return submission

        if sample["prompter_type"] == "livecodebench":
            """
            sample = {
                "selected_uts": {
                    "input_output": {
                        "inputs": List,
                        "outputs: List,
                        "fn_name": str or None,
                    }
                }
            }
            """
            selected_uts = json.loads(sample["selected_uts"])
            input_output = json.loads(selected_uts["input_output"])
            
            assert len(input_output["inputs"]) == len(input_output["outputs"])
            uts = list(zip(input_output["inputs"], input_output["outputs"]))
            
            if len(uts) > sample_size:
                uts = random.sample(uts, sample_size)
            
            fn_name = input_output["fn_name"]
            if fn_name is not None:
                def create_function_call_str(func_name, args_list):
                    args_str = ", ".join(repr(arg) for arg in args_list)
                    return f"{func_name}({args_str})"
                
                if self.run_all_cases:
                    for stdin, stdout in uts:
                        suffix = f"solution = Solution()\nassert {create_function_call_str(fn_name, stdin)} == {repr(stdout)}"
                        submissions.append(build_submission(code_str + "\n" + suffix))
                else:
                    stdin, stdout = uts[0]
                    suffix = f"solution = Solution()\nassert {create_function_call_str(fn_name, stdin)} == {repr(stdout)}"
                    submissions.append(build_submission(code_str + "\n" + suffix))
            else:
                if self.run_all_cases:
                    for stdin, stdout in uts:
                        submissions.append(build_submission(code_str, stdout, stdin))
                else:
                    stdin, stdout = uts[0]
                    submissions.append(build_submission(code_str, stdout, stdin))

        
        elif sample["prompter_type"] == "leetcode":
            try:
                # 确保 input_output 不为 None
                if sample["input_output"] is None:
                    raise RuntimeError("input_output is None")

                if isinstance(sample["input_output"], str):
                    try:
                        input_output = sample["input_output"]
                        while isinstance(input_output, str):
                            try:
                                input_output = json.loads(input_output)
                            except json.JSONDecodeError:
                                break
                    except Exception as e:
                        print(f"JSON解析错误: {e}")
                        raise RuntimeError(f"Failed to parse input_output as JSON: {e}")
                else:
                    input_output = sample["input_output"]

                # 确保 input_output 是字典类型
                if not isinstance(input_output, dict):
                    raise RuntimeError(f"input_output must be a dict, got {type(input_output)}")


                # 确保存在 test_cases 字段
                if "test_cases" not in input_output:
                    raise RuntimeError("test_cases field not found in input_output")

                test_cases = input_output["test_cases"]
                if not test_cases:
                    raise RuntimeError("Empty test cases in leetcode sample")
                
                sample_size = min(sample_size, len(test_cases))
                test_cases = random.sample(test_cases, sample_size)
                
                # 添加提交
                for test_case in test_cases:
                    if not isinstance(test_case, str):
                        print(f"Warning: test_case is not string, converting to string: {test_case}")
                        test_case = str(test_case)
                    submissions.append(build_submission(code_str + "\n" + test_case))
                
            except Exception as e:
                print(f"Error processing leetcode sample: {e}")
                raise RuntimeError(f"Failed to process leetcode sample: {e}")

        
        else:

            # 尝试解析input_output字段，支持字符串格式和对象格式
            if isinstance(sample["input_output"], str):
                try:
                    uts = json.loads(sample["input_output"])
                except json.JSONDecodeError:
                    raise RuntimeError("Invalid input_output format!")
            else:
                uts = sample["input_output"]
            
            if self.run_all_cases:
                
                # 处理包含inputs和outputs的情况
                if isinstance(uts, dict) and "inputs" in uts and "outputs" in uts:
                    inputs = uts["inputs"]
                    outputs = uts["outputs"]
                    if not inputs or not outputs:
                        raise RuntimeError("Empty inputs or outputs in sample")
                else:
                    uts = json.loads(uts)
                    inputs = uts["inputs"]
                    outputs = uts["outputs"]
                assert len(inputs) == len(outputs), "输入和输出数量不匹配"
                
                if len(inputs) > sample_size:
                    indices = random.sample(range(len(inputs)), sample_size)
                    inputs = [inputs[i] for i in indices]
                    outputs = [outputs[i] for i in indices]
                
                if sample["type"] == "stdin_stdout":
                    for input_data, output_data in zip(inputs, outputs):
                        submissions.append(build_submission(code_str, str(output_data), str(input_data)))
                else:
                    raise RuntimeError("Invalid sample type!")
            else:
                raise RuntimeError("We only support run_all_cases")


        return submissions            
                    

    def submit_batch(self, submissions: list) -> list:
        data = {
            "type": "batch",
            "submissions": submissions
        }

        def write_data_to_json(file_path, data):
            try:
                with open(file_path, 'a') as f:
                    # 获取文件锁
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(data, f, indent=4)
                        f.write('\n')  # 添加换行符以分隔不同的记录
                    finally:
                        # 释放文件锁
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except IOError as e:
                print(f"Failed to write to file: {e}")

        try:
            response = requests.post(self.config.reward_model.url, json=data)
            response.raise_for_status()

            results = response.json()['results']
            success_list = [res['success'] for res in results]
            assert len(success_list) == len(submissions)
            return success_list
        except requests.exceptions.RequestException as e:
            # file_path = "/home/superbench/xinzhang3/haoling/epicoder2/submissions.json"
            # write_data_to_json(file_path, data)
            print(f"Request failed: {e}")
            return [False] * len(submissions)
        except (ValueError, KeyError, AssertionError) as e:
            print(f"Failed to process response: {e}")
            return [False] * len(submissions)


    
