#!/usr/bin/env python3
"""
gen_answer_batched.py - 为竞赛编程问题生成答案

该脚本从输入的JSONL文件中读取问题，使用SGlang批量API生成答案，并保存到输出文件。
采用批处理架构。
"""

import os
import json
import random
import numpy as np
import time
import uuid
import requests
import threading
from typing import Dict, List
import argparse

def sort_jsonl_file(file_path):
    """
    排序JSONL文件，确保使用相同的逻辑进行文件排序
    """
    try:
        # 读取JSONL文件中的所有行
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析JSON行并按索引排序
        data = []
        for line in lines:
            if line.strip():  # 跳过空行
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    print(f"警告: 跳过无效的JSON行: {line[:50]}... (错误: {e})")
        
        data.sort(key=lambda x: x.get('question_index', x.get('config_index', float('inf'))))  # 按索引排序
        
        # 将排序后的数据写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in data:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"文件 {file_path} 已排序。共处理 {len(data)} 条记录。")
    except Exception as e:
        print(f"排序文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()

# 线程锁用于文件写入
lock = threading.Lock()

# Worker IP 管理器
class WorkerIPManager:
    def __init__(self, worker_ips=None, update_interval=300):
        """
        初始化 Worker IP 管理器
        """
        self.worker_ips = worker_ips or ["localhost"]  # 默认使用本地
        self._ips_lock = threading.Lock()
        self.update_interval = update_interval
        print(f"使用固定的 Worker IPs: {self.worker_ips}")

    def get_cached_ips(self):
        with self._ips_lock:
            return list(self.worker_ips)

    def stop_update_thread(self):
        """保持接口一致性"""
        pass

def sglang_call(prompt, worker_ip, port, model_name, max_new_tokens=32768, temperature=0.6, top_p=0.95, top_k=20, min_p=0):
    """
    使用 SGlang 调用 LLM
    """
    url = f"http://{worker_ip}:{port}/v1/chat/completions"
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional competitive programming expert and code implementer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p
    }
    
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer None',  # SGlang 通常不需要真实的 token
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        res = response.json()

        if 'choices' in res and len(res['choices']) > 0:
            content = res['choices'][0]['message']['content']
            return content
        else:
            print(f"意外的响应结构: {res}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解码失败: {str(e)}")
        return None
    except Exception as e:
        print(f"调用LLM时发生意外错误: {str(e)}")
        return None

def load_questions_from_jsonl(file_path):
    """
    从JSONL文件中加载问题
    """
    questions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                        # 提取问题内容
                        question_content = None
                        question_index = line_num - 1  # 默认使用行号作为索引
                        
                        # 尝试多种可能的问题字段名
                        for field in ['question', 'problem_statement', 'task', 'content']:
                            if field in record:
                                question_content = record[field]
                                break
                        
                        # 尝试提取索引
                        for index_field in ['question_index', 'config_index', 'index', 'id']:
                            if index_field in record:
                                question_index = record[index_field]
                                break
                        
                        if question_content:
                            questions.append({
                                'question_index': question_index,
                                'question_content': question_content,
                                'original_record': record
                            })
                        else:
                            print(f"警告: 行 {line_num} 中未找到问题内容")
                    except json.JSONDecodeError as e:
                        print(f"警告: 行 {line_num} JSON解析失败: {e}")
        
        print(f"成功加载 {len(questions)} 个问题")
        return questions
        
    except FileNotFoundError:
        print(f"错误: 输入文件 '{file_path}' 未找到")
        return []
    except Exception as e:
        print(f"加载问题文件时出错: {e}")
        return []

def generate_answer_prompt(question_content):
    """
    生成答案生成的prompt
    """
    return f"""Now that you are a code expert, I have provided you with the QUESTION. Complete the problem with awesome code logic and give a richly commented analysis in the code of your answer. Include the necessary packages. Give out code implementation. Enclose the python code with ```python and ```
- QUESTION {question_content} -
"""

def process_single_question(question_data, worker_ip, port, model_name, output_path, output_lock, verbose=False):
    """
    处理单个问题，生成答案
    """
    question_index = question_data['question_index']
    question_content = question_data['question_content']
    original_record = question_data['original_record']
    
    try:
        if verbose:
            print(f"开始处理问题 {question_index}...")
        
        # 生成prompt
        prompt = generate_answer_prompt(question_content)
        
        if verbose:
            print(f"问题 {question_index} prompt长度: {len(prompt)}")
        
        # 调用API生成答案
        result = sglang_call(prompt, worker_ip, port, model_name, temperature=0.6)
        
        if result is None:
            error_result = {
                "question_index": question_index,
                "error": "Failed to generate answer",
                "question_content": question_content[:200] + "..." if len(question_content) > 200 else question_content,
                "original_record": original_record
            }
            
            with output_lock:
                with open(output_path, 'a', encoding='utf-8') as f:
                    json.dump(error_result, f, ensure_ascii=False)
                    f.write('\n')
            return error_result
        
        # 解析答案
        try:
            # 提取Python代码块
            python_code = ""
            if "```python" in result:
                start_idx = result.find("```python")
                end_idx = result.find("```", start_idx + 9)
                if end_idx != -1:
                    python_code = result[start_idx + 9:end_idx].strip()
            
            answer_result = {
                "question_index": question_index,
                "question_content": question_content,
                "generated_answer": result,
                "extracted_code": python_code,
                "original_record": original_record,
                "generation_metadata": {
                    "model_name": model_name,
                    "worker_ip": worker_ip,
                    "timestamp": time.time()
                }
            }
            
            # 线程安全地写入输出文件
            with output_lock:
                with open(output_path, 'a', encoding='utf-8') as f:
                    json.dump(answer_result, f, ensure_ascii=False)
                    f.write('\n')
                if verbose:
                    print(f"已保存问题 {question_index} 的答案")
            
            return answer_result
            
        except Exception as e:
            error_result = {
                "question_index": question_index,
                "error": f"Answer processing error: {str(e)}",
                "raw_result": result,
                "question_content": question_content,
                "original_record": original_record
            }
            
            with output_lock:
                with open(output_path, 'a', encoding='utf-8') as f:
                    json.dump(error_result, f, ensure_ascii=False)
                    f.write('\n')
            
            return error_result
            
    except Exception as e:
        print(f"处理问题 {question_index} 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        error_result = {
            "question_index": question_index,
            "error": f"Processing error: {str(e)}",
            "question_content": question_content,
            "original_record": original_record
        }
        
        with output_lock:
            with open(output_path, 'a', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False)
                f.write('\n')
        
        return error_result

def create_sglang_batch(questions_batch, indices_batch, worker_ip, port, model_name):
    """
    创建 SGlang Batches API 任务用于答案生成
    """
    import tempfile
    import json
    from openai import OpenAI
    
    # 准备批量请求
    batch_requests = []
    for question_data, index in zip(questions_batch, indices_batch):
        try:
            question_content = question_data['question_content']
            
            # 生成prompt
            prompt = generate_answer_prompt(question_content)
            
            # 创建单个请求
            request = {
                "custom_id": f"question_{question_data['question_index']}",  # 使用实际的question_index
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a professional competitive programming expert and code implementer."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 32768,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0
                }
            }
            batch_requests.append(request)
        except Exception as e:
            print(f"准备批量请求 {question_data.get('question_index', index)} 时出错: {e}")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in batch_requests:
            f.write(json.dumps(req) + '\n')
        temp_file_path = f.name
    
    # 使用 OpenAI 客户端创建批量任务
    client = OpenAI(base_url=f"http://{worker_ip}:{port}/v1", api_key="None")
    
    try:
        # 上传文件
        with open(temp_file_path, 'rb') as f:
            uploaded_file = client.files.create(file=f, purpose="batch")
        
        # 创建批量任务
        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        return batch_job, uploaded_file.id, temp_file_path
        
    except Exception as e:
        print(f"创建 SGlang 批量任务失败: {e}")
        # 清理临时文件
        import os
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return None, None, None

def wait_for_batch_completion(batch_job, client, max_wait_time=3600):
    """
    等待批量任务完成
    """
    import time
    
    start_time = time.time()
    while batch_job.status not in ["completed", "failed", "cancelled"]:
        if time.time() - start_time > max_wait_time:
            print(f"批量任务超时 ({max_wait_time}s)")
            return None
        
        time.sleep(10)  # 每10秒检查一次
        try:
            batch_job = client.batches.retrieve(batch_job.id)
            # 安全地访问 request_counts
            if hasattr(batch_job, 'request_counts') and batch_job.request_counts is not None:
                completed = getattr(batch_job.request_counts, 'completed', 0)
                total = getattr(batch_job.request_counts, 'total', 0)
                print(f"批量任务状态: {batch_job.status}, 已完成: {completed}/{total}")
            else:
                print(f"批量任务状态: {batch_job.status}")
        except Exception as e:
            print(f"检查批量任务状态时出错: {e}")
            # 如果获取状态失败，继续等待
            continue
    
    return batch_job

def process_batch_results(completed_batch, client, questions_batch, output_path, output_lock, verbose=False):
    """
    处理批量任务的结果
    """
    import json
    
    # 获取结果
    result_file_id = completed_batch.output_file_id
    if not result_file_id:
        raise Exception("批量任务完成但没有输出文件")
        
    file_response = client.files.content(result_file_id)
    result_content = file_response.read().decode("utf-8")
    
    results = [
        json.loads(line) for line in result_content.split("\n") if line.strip() != ""
    ]
    
    # 创建索引映射
    question_map = {f"question_{q['question_index']}": q for q in questions_batch}
    
    processed_count = 0
    
    for result in results:
        custom_id = result['custom_id']
        if custom_id not in question_map:
            print(f"警告: 未找到对应的问题 {custom_id}")
            continue
            
        question_data = question_map[custom_id]
        question_index = question_data['question_index']
        question_content = question_data['question_content']
        original_record = question_data['original_record']
        
        try:
            if verbose:
                print(f"处理批量结果: 问题 {question_index}")
            
            if result['response']['status_code'] != 200:
                error_result = {
                    "question_index": question_index,
                    "error": f"Batch API error: {result['response']}",
                    "question_content": question_content[:200] + "..." if len(question_content) > 200 else question_content,
                    "original_record": original_record,
                    "batch_mode": True
                }
                
                with output_lock:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        json.dump(error_result, f, ensure_ascii=False)
                        f.write('\n')
                continue
            
            # 从批量结果中提取内容
            response_body = result['response']['body']
            
            # 安全地提取内容，处理不同的 API 响应格式
            choices = response_body.get('choices')
            if isinstance(choices, list) and len(choices) > 0:
                # 标准 OpenAI 格式: choices 是列表
                content = choices[0].get('message', {}).get('content', '')
            elif isinstance(choices, dict):
                # SGlang 批量格式: choices 是字典
                content = choices.get('message', {}).get('content', '')
            else:
                print(f"无法解析响应格式: {response_body}")
                error_result = {
                    "question_index": question_index,
                    "error": f"Invalid response format: {response_body}",
                    "question_content": question_content[:200] + "..." if len(question_content) > 200 else question_content,
                    "original_record": original_record,
                    "batch_mode": True
                }
                
                with output_lock:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        json.dump(error_result, f, ensure_ascii=False)
                        f.write('\n')
                continue
            
            # 提取Python代码块
            python_code = ""
            if "```python" in content:
                start_idx = content.find("```python")
                end_idx = content.find("```", start_idx + 9)
                if end_idx != -1:
                    python_code = content[start_idx + 9:end_idx].strip()
            
            answer_result = {
                "question_index": question_index,
                "question_content": question_content,
                "generated_answer": content,
                "extracted_code": python_code,
                "original_record": original_record,
                "generation_metadata": {
                    "model_name": "batch_processed",
                    "timestamp": time.time(),
                    "batch_mode": True
                }
            }
            
            # 线程安全地写入输出文件
            with output_lock:
                with open(output_path, 'a', encoding='utf-8') as f:
                    json.dump(answer_result, f, ensure_ascii=False)
                    f.write('\n')
                if verbose:
                    print(f"已保存批量处理的问题 {question_index} 答案")
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理批量结果 {question_index} 时出错: {e}")
            error_result = {
                "question_index": question_index,
                "error": f"Processing batch result error: {str(e)}",
                "question_content": question_content[:200] + "..." if len(question_content) > 200 else question_content,
                "original_record": original_record,
                "batch_mode": True
            }
            
            with output_lock:
                with open(output_path, 'a', encoding='utf-8') as f:
                    json.dump(error_result, f, ensure_ascii=False)
                    f.write('\n')
    
    return processed_count

def cleanup_batch_files(client, uploaded_file_id, result_file_id, temp_file_path):
    """
    清理批量任务相关文件
    """
    import os
    
    cleanup_errors = []
    
    # 清理上传的文件
    if uploaded_file_id:
        try:
            client.files.delete(uploaded_file_id)
        except Exception as e:
            cleanup_errors.append(f"删除上传文件失败: {e}")
    
    # 清理结果文件
    if result_file_id:
        try:
            client.files.delete(result_file_id)
        except Exception as e:
            cleanup_errors.append(f"删除结果文件失败: {e}")
    
    # 清理临时文件
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except Exception as e:
            cleanup_errors.append(f"删除临时文件失败: {e}")
    
    if cleanup_errors:
        print(f"清理文件时出现一些错误: {'; '.join(cleanup_errors)}")

def process_questions_batch(questions_batch, worker_ip, port, model_name, output_path, output_lock, 
                           use_batch_api=False, verbose=False):
    """
    处理一批问题，支持批处理和单个处理两种模式
    """
    worker_id_str = f"Worker-{worker_ip}:{port}"
    
    if use_batch_api and len(questions_batch) > 1:
        # 使用批处理模式
        print(f"[{worker_id_str}] 使用批处理模式处理 {len(questions_batch)} 个问题...")
        
        from openai import OpenAI
        client = OpenAI(base_url=f"http://{worker_ip}:{port}/v1", api_key="None")
        
        # 最大重试次数
        max_retries = 2
        retry_delay = 5  # 重试间隔（秒）
        
        for retry_count in range(max_retries + 1):
            try:
                # 创建批量任务
                indices_batch = [q['question_index'] for q in questions_batch]
                batch_job, uploaded_file_id, temp_file_path = create_sglang_batch(
                    questions_batch, indices_batch, worker_ip, port, model_name
                )
                
                if batch_job is None:
                    if retry_count < max_retries:
                        print(f"[{worker_id_str}] 批量任务创建失败，{retry_delay}秒后重试 ({retry_count + 1}/{max_retries + 1})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"[{worker_id_str}] 批量任务创建多次失败，回退到单个处理模式")
                        return process_questions_single(questions_batch, worker_ip, port, model_name, output_path, output_lock, verbose)
                break
                
            except Exception as e:
                if retry_count < max_retries:
                    print(f"[{worker_id_str}] 创建批量任务时出错: {e}，{retry_delay}秒后重试 ({retry_count + 1}/{max_retries + 1})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[{worker_id_str}] 创建批量任务多次失败: {e}，回退到单个处理模式")
                    return process_questions_single(questions_batch, worker_ip, port, model_name, output_path, output_lock, verbose)
        
        print(f"[{worker_id_str}] 批量任务已创建: {batch_job.id}")
        
        # 等待批量任务完成，增加重试机制
        completed_batch = None
        for wait_retry in range(2):  # 最多重试等待2次
            try:
                completed_batch = wait_for_batch_completion(batch_job, client)
                if completed_batch is not None:
                    break
            except Exception as e:
                print(f"[{worker_id_str}] 等待批量任务完成时出错: {e}")
                if wait_retry < 1:
                    print(f"[{worker_id_str}] 重试等待批量任务...")
                    time.sleep(retry_delay)
                    
        if completed_batch is None:
            print(f"[{worker_id_str}] 批量任务超时或失败")
            cleanup_batch_files(client, uploaded_file_id, None, temp_file_path)
            return process_questions_single(questions_batch, worker_ip, port, model_name, output_path, output_lock, verbose)
        
        if completed_batch.status != "completed":
            print(f"[{worker_id_str}] 批量任务失败，状态: {completed_batch.status}")
            cleanup_batch_files(client, uploaded_file_id, None, temp_file_path)
            return process_questions_single(questions_batch, worker_ip, port, model_name, output_path, output_lock, verbose)
        
        print(f"[{worker_id_str}] 批量任务完成，开始处理结果")
        
        # 处理批量结果
        try:
            processed_count = process_batch_results(
                completed_batch, client, questions_batch, output_path, output_lock, verbose
            )
            
            # 清理文件
            cleanup_batch_files(client, uploaded_file_id, completed_batch.output_file_id, temp_file_path)
            
            print(f"[{worker_id_str}] 批处理完成，成功处理 {processed_count} 个问题")
            return processed_count
            
        except Exception as e:
            print(f"[{worker_id_str}] 处理批量结果时出错: {e}")
            cleanup_batch_files(client, uploaded_file_id, getattr(completed_batch, 'output_file_id', None), temp_file_path)
            return process_questions_single(questions_batch, worker_ip, port, model_name, output_path, output_lock, verbose)
            
    else:
        # 使用单个处理模式
        if verbose:
            print(f"[{worker_id_str}] 使用单个处理模式处理 {len(questions_batch)} 个问题...")
        return process_questions_single(questions_batch, worker_ip, port, model_name, output_path, output_lock, verbose)

def process_questions_single(questions_batch, worker_ip, port, model_name, output_path, output_lock, verbose=False):
    """
    单个处理模式
    """
    worker_id_str = f"Worker-{worker_ip}:{port}"
    processed_count = 0
    
    for question_data in questions_batch:
        try:
            result = process_single_question(
                question_data, worker_ip, port, model_name, output_path, output_lock, verbose
            )
            if result and "error" not in result:
                processed_count += 1
        except Exception as e:
            print(f"[{worker_id_str}] 处理问题 {question_data.get('question_index', 'unknown')} 时出错: {e}")
    
    print(f"[{worker_id_str}] 单个处理模式完成，处理了 {processed_count} 个问题")
    return processed_count

def main():
    parser = argparse.ArgumentParser(description='为竞赛编程问题生成答案')
    parser.add_argument('--input_file', type=str, required=True,
                       help='输入的问题JSONL文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出的答案JSONL文件路径')
    parser.add_argument('--worker_ips', type=str, nargs='+', 
                       default=["localhost"], 
                       help='Worker IP 列表')
    parser.add_argument('--worker_port', type=int, default=30000, help='Worker端口')
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-0528", 
                       help='模型名称')
    parser.add_argument('--batch_size', type=int, default=16, help='每批处理的任务数量')
    parser.add_argument('--begin_idx', type=int, default=0, help='起始索引')
    parser.add_argument('--end_idx', type=int, default=-1, help='结束索引 (-1表示处理全部)')
    parser.add_argument('--use_batch_api', action='store_true', 
                       help='使用SGlang Batches API进行批量处理')
    parser.add_argument('--verbose', action='store_true', help='打印详细输出')
    
    args = parser.parse_args()
    
    # 初始化 Worker IP 管理器
    ip_manager = WorkerIPManager(worker_ips=args.worker_ips)
    
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载问题
    all_questions = load_questions_from_jsonl(args.input_file)
    if not all_questions:
        print("没有加载到任何问题，退出")
        return

    # 统一按 question_index 排序，确保处理顺序稳定
    all_questions.sort(key=lambda q: q['question_index'])

    total_questions = len(all_questions)
    begin_idx = args.begin_idx
    end_idx = args.end_idx

    if end_idx == -1:
        # 处理从指定 question_index 起的所有问题
        selected_questions = [
            q for q in all_questions
            if q['question_index'] >= begin_idx
        ]
    else:
        selected_questions = [
            q for q in all_questions
            if begin_idx <= q['question_index'] <= end_idx
        ]

    if not selected_questions:
        # 若根据 question_index 无匹配，则回退到旧的基于位置的切片逻辑以保持兼容
        if end_idx == -1:
            positional_selected = all_questions[begin_idx:] if begin_idx < total_questions else []
        elif end_idx >= begin_idx:
            slice_end = min(end_idx + 1, total_questions)
            positional_selected = all_questions[begin_idx:slice_end]
        else:
            positional_selected = []

        if positional_selected:
            print("根据 question_index 范围未选中任何问题，已回退到位置切片方式。")
            selected_questions = positional_selected

    if selected_questions:
        first_q_index = selected_questions[0]['question_index']
        last_q_index = selected_questions[-1]['question_index']
        print(f"从 {total_questions} 个问题中选择了 {len(selected_questions)} 个 (question_index {first_q_index} 到 {last_q_index})")
    else:
        end_desc = f"{end_idx}" if end_idx != -1 else "末尾"
        print(f"从 {total_questions} 个问题中选择了 0 个 (请求的 question_index 区间 {begin_idx} 到 {end_desc})")

    # 创建线程锁用于输出文件写入
    output_lock = threading.Lock()
    
    # 在开始处理前，确保输出文件存在
    with open(args.output_file, 'w', encoding='utf-8') as f:
        pass  # 创建空文件
    
    # 获取可用的 Worker IPs
    available_ips = ip_manager.get_cached_ips()
    if not available_ips:
        print("错误: 没有可用的 Worker IP，无法处理任务。")
        return
    
    print(f"使用 Worker IP: {available_ips[0]}")
    
    # 处理所有问题
    worker_ip = available_ips[0]
    total_processed = 0
    
    print(f"开始处理 {len(selected_questions)} 个问题...")
    print(f"批处理模式: {'启用' if args.use_batch_api else '禁用'}")
    
    # 按批次处理问题
    total_batches = (len(selected_questions) + args.batch_size - 1) // args.batch_size
    
    for batch_start in range(0, len(selected_questions), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(selected_questions))
        questions_batch = selected_questions[batch_start:batch_end]
        batch_num = batch_start // args.batch_size + 1
        
        print(f"处理批次 {batch_num}/{total_batches}: 问题 {batch_start} - {batch_end-1} ({len(questions_batch)} 个问题)")
        
        # 处理当前批次
        processed_count = process_questions_batch(
            questions_batch, worker_ip, args.worker_port, args.model_name,
            args.output_file, output_lock, args.use_batch_api, args.verbose
        )
        
        total_processed += processed_count
        print(f"批次 {batch_num} 完成，已处理: {total_processed}/{len(selected_questions)}")
        
        # 定期排序文件
        if batch_end % 50 == 0:
            sort_jsonl_file(args.output_file)
    
    print(f"所有问题处理完成，总共处理了 {total_processed} 个答案")
    
    # 最终排序输出文件
    sort_jsonl_file(args.output_file)
    
    # 停止 IP 更新线程
    ip_manager.stop_update_thread()
    print("所有任务处理完成。")

if __name__ == "__main__":
    main()

# 示例命令:
# 使用批处理API
# python gen_answer_batched.py \
# --input_file v3_results/examples_with_feature_purpose42-step0-10000.jsonl \
# --output_file v3_results/answer_with_feature_purpose42-step0-10000.jsonl \
# --worker_ips localhost \
# --worker_port 30000 \
# --model_name Qwen3-235B-A22B-Thinking-2507 \
# --batch_size 256 \
# --use_batch_api \
# --verbose
