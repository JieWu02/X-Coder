#!/usr/bin/env python3
"""
多数投票准确率统计实验 - 主脚本（并行版本）
对 TACO-verified_500_sampled.jsonl 进行多数投票准确率统计
"""
import json
import os
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from code_executor import CodeExecutor
from majority_voting import MajorityVoter


# ==================== 配置 ====================
CONFIG = {
    'input_file': 'TACO-verified_500_sampled.jsonl',
    'output_file': 'voting_results.jsonl',
    'checkpoint_file': 'voting_checkpoint.json',
    'log_file': 'voting_experiment.log',

    'num_codes_to_use': 8,        # 使用前8条 sampled_solutions
    'timeout': 5,                  # 代码执行超时（秒）
    'save_interval': 10,           # 每处理多少题保存一次
    'num_workers': min(64, cpu_count()),  # 并行工作进程数
}


# ==================== 日志 ====================
def log(message, level='INFO'):
    """写入日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f'[{timestamp}] [{level}] {message}'
    print(log_message)

    with open(CONFIG['log_file'], 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')


# ==================== 检查点 ====================
def load_checkpoint():
    """加载检查点"""
    if os.path.exists(CONFIG['checkpoint_file']):
        try:
            with open(CONFIG['checkpoint_file'], 'r') as f:
                checkpoint = json.load(f)
                log(f"加载检查点：已处理 {checkpoint.get('processed_count', 0)} 题")
                return checkpoint
        except Exception as e:
            log(f"加载检查点失败: {e}", 'ERROR')
    return {'processed_ids': [], 'processed_count': 0}


def save_checkpoint(checkpoint):
    """保存检查点"""
    try:
        with open(CONFIG['checkpoint_file'], 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        log(f"保存检查点失败: {e}", 'ERROR')


# ==================== 处理单个问题 ====================
def process_question(item):
    """
    处理单个问题，对所有 test case 进行多数投票
    （独立函数，用于并行处理）

    Args:
        item: 数据项

    Returns:
        dict: 结果
    """
    # 在每个进程内创建 executor 和 voter
    executor = CodeExecutor(timeout=CONFIG['timeout'])
    voter = MajorityVoter(tie_break_strategy='random')

    question_id = item['id']
    difficulty = item['difficulty']
    source = item['source']

    # 获取前8条 sampled_solutions
    sampled_solutions = item['sampled_solutions'][:CONFIG['num_codes_to_use']]
    codes = [sol['code'] for sol in sampled_solutions]

    # 解析 input_output
    io_data = json.loads(item['input_output'])
    test_inputs = io_data['inputs']
    test_outputs = io_data['outputs']

    # 对每个 test case 进行投票
    test_results = []
    correct_count = 0

    for test_idx, (test_input, expected_output) in enumerate(zip(test_inputs, test_outputs)):
        # 规范化 expected_output（可能是字符串或列表）
        if isinstance(expected_output, list):
            expected_output_str = json.dumps(expected_output, ensure_ascii=False)
        else:
            expected_output_str = str(expected_output)

        # 执行所有代码
        execution_results = []
        for code_idx, code in enumerate(codes):
            exec_result = executor.execute(code, test_input)
            execution_results.append({
                'code_index': code_idx,
                'success': exec_result['success'],
                'output': exec_result['output'],
                'error': exec_result['error'] if not exec_result['success'] else None,
                'timeout': exec_result.get('timeout', False)
            })

        # 多数投票
        vote_result = voter.vote(execution_results)

        # 比较结果
        voted_output = vote_result['voted_output']
        is_correct = voter.compare_with_expected(voted_output, expected_output_str)

        if is_correct:
            correct_count += 1

        # 记录结果
        test_results.append({
            'test_index': test_idx,
            'test_input': test_input[:100] + '...' if len(test_input) > 100 else test_input,  # 截断
            'expected_output': expected_output_str.strip(),
            'voted_output': voted_output.strip() if voted_output else None,
            'is_correct': is_correct,
            'vote_counts': vote_result['vote_counts'],
            'success_rate': vote_result['success_rate'],
            'total_votes': vote_result['total_votes'],
            'has_tie': vote_result['has_tie'],
            'execution_summary': {
                'success': sum(1 for r in execution_results if r['success']),
                'timeout': sum(1 for r in execution_results if r.get('timeout', False)),
                'error': sum(1 for r in execution_results if not r['success'] and not r.get('timeout', False))
            }
        })

    # 计算整体准确率
    total_tests = len(test_results)
    overall_accuracy = correct_count / total_tests if total_tests > 0 else 0.0

    # 计算平均成功率
    avg_success_rate = sum(t['success_rate'] for t in test_results) / total_tests if total_tests > 0 else 0.0

    return {
        'question_id': question_id,
        'difficulty': difficulty,
        'source': source,
        'total_test_cases': total_tests,
        'correct_predictions': correct_count,
        'overall_accuracy': overall_accuracy,
        'avg_code_success_rate': avg_success_rate,
        'test_results': test_results
    }


# ==================== 主函数 ====================
def main():
    log("=" * 70)
    log("开始多数投票准确率统计实验（并行版本）")
    log("=" * 70)
    log(f"配置: {CONFIG}")
    log(f"并行进程数: {CONFIG['num_workers']}")

    # 加载检查点
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint.get('processed_ids', []))

    # 读取数据
    log(f"读取数据: {CONFIG['input_file']}")
    items = []
    with open(CONFIG['input_file'], 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item['id'] not in processed_ids:
                items.append(item)

    total_items = len(items)
    log(f"待处理问题数: {total_items}")

    if total_items == 0:
        log("没有待处理的问题")
        return

    # 如果是首次运行，清空输出文件
    if len(processed_ids) == 0:
        with open(CONFIG['output_file'], 'w', encoding='utf-8') as f:
            pass
        log(f"初始化输出文件: {CONFIG['output_file']}")

    # 使用多进程并行处理
    results_buffer = []
    processed_count = checkpoint.get('processed_count', 0)

    log(f"开始并行处理...")

    with Pool(processes=CONFIG['num_workers']) as pool:
        # 使用 imap_unordered 以便结果完成就立即处理
        for result in tqdm(
            pool.imap_unordered(process_question, items),
            total=total_items,
            desc="处理问题",
            ncols=100
        ):
            try:
                results_buffer.append(result)
                processed_ids.add(result['question_id'])
                processed_count += 1

                # 定期保存
                if len(results_buffer) >= CONFIG['save_interval']:
                    # 保存结果
                    with open(CONFIG['output_file'], 'a', encoding='utf-8') as f:
                        for res in results_buffer:
                            f.write(json.dumps(res, ensure_ascii=False) + '\n')

                    # 更新检查点
                    checkpoint['processed_ids'] = list(processed_ids)
                    checkpoint['processed_count'] = processed_count
                    save_checkpoint(checkpoint)

                    log(f"已处理 {processed_count}/{total_items} 题，已保存")
                    results_buffer = []

            except Exception as e:
                log(f"处理结果时出错: {e}", 'ERROR')
                import traceback
                log(traceback.format_exc(), 'ERROR')

    # 保存剩余结果
    if results_buffer:
        with open(CONFIG['output_file'], 'a', encoding='utf-8') as f:
            for res in results_buffer:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

        checkpoint['processed_ids'] = list(processed_ids)
        checkpoint['processed_count'] = processed_count
        save_checkpoint(checkpoint)

    log("=" * 70)
    log("实验完成！")
    log("=" * 70)
    log(f"处理问题数: {processed_count}")
    log(f"结果文件: {CONFIG['output_file']}")
    log("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log("\n收到中断信号，保存进度...", 'WARNING')
        log("可以稍后重新运行脚本继续处理", 'INFO')
    except Exception as e:
        log(f"程序异常退出: {e}", 'ERROR')
        import traceback
        log(traceback.format_exc(), 'ERROR')
