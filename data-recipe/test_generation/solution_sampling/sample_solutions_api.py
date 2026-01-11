#!/usr/bin/env python3
"""
使用 OpenAI-compatible API 对 TACO-verified 数据集进行多线程采样
- 每道题采样 16 条答案
- 64 线程并发
- 每 10 个问题保存一次
- 支持断点续传
- 目标处理 1300 个问题
"""
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from openai_api_client import GPTClient

# ==================== 配置区域 ====================
CONFIG = {
    "input_file": "TACO-verified_filtered_testcase_ge_8.jsonl",
    "output_file": "TACO-verified_with_samples.jsonl",
    "checkpoint_file": "api_sampling_checkpoint.json",
    "log_file": "api_sampling.log",

    "target_count": 1300,       # 目标处理数量
    "samples_per_question": 20, # 每题采样数
    "max_workers": 64,          # 线程数
    "save_interval": 10,        # 每处理多少个保存一次

    "reasoning_effort": "medium",  # 保留字段（OpenAI 兼容接口不强制使用）
    "max_completion_tokens": 32768, # 最大 token 数 (修改为32768)
    "timeout": 500,                # 超时时间（秒）(修改为500)
}

PROMPT_TEMPLATE = """Now that you are a code expert, I have provided you with the QUESTION. Complete the problem with excellent code logic and include richly commented analysis within the code. Import all necessary packages, provide the full code implementation, and enclose the Python code within ```python ``` .

– QUESTION
{question_content}
–"""

# ==================== 全局变量 ====================
processed_count = 0
success_count = 0
failed_count = 0
lock = threading.Lock()
gpt_client = None

# ==================== 日志函数 ====================
def log(message, level="INFO"):
    """写入日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)

    with open(CONFIG["log_file"], "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


# ==================== 检查点管理 ====================
def load_checkpoint():
    """加载检查点"""
    if os.path.exists(CONFIG["checkpoint_file"]):
        try:
            with open(CONFIG["checkpoint_file"], "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                log(f"加载检查点：已处理 {checkpoint.get('processed_count', 0)} 个问题")
                return checkpoint
        except Exception as e:
            log(f"加载检查点失败: {e}", "ERROR")
    return {"processed_ids": [], "processed_count": 0}


def save_checkpoint(checkpoint):
    """保存检查点"""
    try:
        with open(CONFIG["checkpoint_file"], "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        log(f"保存检查点失败: {e}", "ERROR")


# ==================== 采样函数 ====================
def sample_single_answer(question, question_id, sample_idx):
    """为单个问题采样一个答案"""
    global gpt_client

    # 构造 prompt
    prompt = PROMPT_TEMPLATE.format(question_content=question)

    try:
        # 调用 OpenAI-compatible API
        content, usage = gpt_client.call_chat(
            prompt=prompt,
            reasoning_effort=CONFIG["reasoning_effort"],
            max_completion_tokens=CONFIG["max_completion_tokens"],
            timeout=CONFIG["timeout"],
        )

        if content:
            return {
                "success": True,
                "content": content,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                    "reasoning_tokens": 0,
                }
            }
        else:
            return {
                "success": False,
                "error": "API returned no content",
                "content": None
            }

    except Exception as e:
        log(f"问题 {question_id} 样本 {sample_idx} 采样失败: {e}", "ERROR")
        return {
            "success": False,
            "error": str(e),
            "content": None
        }


def process_single_question(item, question_index):
    """处理单个问题（采样16个答案，串行）"""
    global processed_count, success_count, failed_count, lock

    question_id = item.get("id")
    question = item.get("question", "")

    if not question:
        log(f"问题 {question_id} (索引 {question_index}) 没有 question 字段，跳过", "WARNING")
        return None

    log(f"开始处理问题 {question_id} (索引 {question_index})")

    # 采样 16 个答案（串行）
    sampled_answers = []
    sample_results = []

    for i in range(CONFIG["samples_per_question"]):
        result = sample_single_answer(question, question_id, i)
        sample_results.append(result)

        if result["success"]:
            sampled_answers.append(result["content"])
        else:
            sampled_answers.append(f"[ERROR]: {result.get('error', 'Unknown error')}")

        # 短暂延迟避免 API 过载
        time.sleep(0.1)

    # 添加采样结果到原始数据
    item_with_samples = item.copy()
    item_with_samples["sampled_answers"] = sampled_answers
    item_with_samples["sampling_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "prompt_template": PROMPT_TEMPLATE,
        "config": {
            "reasoning_effort": CONFIG["reasoning_effort"],
            "max_completion_tokens": CONFIG["max_completion_tokens"],
            "samples_count": CONFIG["samples_per_question"],
        },
        "results": sample_results,
    }

    # 更新统计
    with lock:
        processed_count += 1
        success_samples = sum(1 for r in sample_results if r["success"])
        if success_samples > 0:
            success_count += 1
        else:
            failed_count += 1

        log(f"完成问题 {question_id}: {success_samples}/{CONFIG['samples_per_question']} 个答案成功")

    return item_with_samples


# ==================== 批量保存 ====================
def save_results(results, mode="a"):
    """保存结果到文件"""
    try:
        with open(CONFIG["output_file"], mode, encoding="utf-8") as f:
            for result in results:
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        log(f"保存了 {len([r for r in results if r is not None])} 个结果")
    except Exception as e:
        log(f"保存结果失败: {e}", "ERROR")


# ==================== 主函数 ====================
def main():
    global gpt_client, processed_count, success_count, failed_count

    log("=" * 70)
    log("开始 OpenAI-compatible API 采样任务")
    log("=" * 70)
    log(f"配置: {CONFIG}")

    # 初始化 GPT 客户端
    log("初始化 OpenAI-compatible API 客户端...")
    gpt_client = GPTClient()
    log("OpenAI-compatible API 客户端初始化完成")

    # 加载检查点
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint.get("processed_ids", []))

    # 读取数据
    log(f"读取输入文件: {CONFIG['input_file']}")
    items = []
    with open(CONFIG["input_file"], "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            # 跳过已处理的
            if item.get("id") not in processed_ids:
                items.append(item)

    log(f"待处理问题数: {len(items)}")

    # 限制处理数量
    items_to_process = items[:CONFIG["target_count"]]
    total_items = len(items_to_process)

    log(f"本次处理 {total_items} 个问题")
    log(f"预计 API 调用次数: {total_items * CONFIG['samples_per_question']}")

    # 如果是首次运行，清空输出文件
    if processed_count == 0 and not processed_ids:
        with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
            pass
        log(f"初始化输出文件: {CONFIG['output_file']}")

    # 多线程处理
    log(f"启动 {CONFIG['max_workers']} 个线程...")

    results_buffer = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_single_question, item, idx): (idx, item)
            for idx, item in enumerate(items_to_process)
        }

        # 处理完成的任务
        with tqdm(total=total_items, desc="处理问题", ncols=100) as pbar:
            for future in as_completed(future_to_index):
                idx, item = future_to_index[future]

                try:
                    result = future.result()

                    if result:
                        results_buffer.append(result)
                        processed_ids.add(result.get("id"))

                    # 每处理 save_interval 个，保存一次
                    if len(results_buffer) >= CONFIG["save_interval"]:
                        save_results(results_buffer, mode="a")

                        # 更新检查点
                        checkpoint["processed_ids"] = list(processed_ids)
                        checkpoint["processed_count"] = processed_count
                        save_checkpoint(checkpoint)

                        results_buffer = []

                except Exception as e:
                    log(f"处理问题 {item.get('id')} (索引 {idx}) 时出错: {e}", "ERROR")

                finally:
                    pbar.update(1)

                    # 更新进度信息
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        "成功": success_count,
                        "失败": failed_count,
                        "速率": f"{rate:.2f} q/s"
                    })

    # 保存剩余结果
    if results_buffer:
        save_results(results_buffer, mode="a")
        checkpoint["processed_ids"] = list(processed_ids)
        checkpoint["processed_count"] = processed_count
        save_checkpoint(checkpoint)

    # 统计信息
    elapsed_time = time.time() - start_time

    log("=" * 70)
    log("采样任务完成！")
    log("=" * 70)
    log(f"总处理数: {processed_count}")
    log(f"成功: {success_count}")
    log(f"失败: {failed_count}")
    log(f"总耗时: {elapsed_time/60:.2f} 分钟")
    log(f"平均速率: {processed_count/elapsed_time:.2f} 问题/秒")
    log(f"结果文件: {CONFIG['output_file']}")
    log("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n收到中断信号，保存进度...", "WARNING")
        log("可以稍后重新运行脚本继续处理", "INFO")
    except Exception as e:
        log(f"程序异常退出: {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
