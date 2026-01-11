#!/usr/bin/env python3
"""
多数投票准确率统计实验 - 可配置版本
支持命令行参数指定使用的solution数量
"""
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from code_executor import CodeExecutor
from majority_voting import MajorityVoter


# ==================== 处理单个问题 ====================
def process_question(item, num_codes):
    """
    处理单个问题，对所有 test case 进行多数投票
    """
    executor = CodeExecutor(timeout=5)
    voter = MajorityVoter(tie_break_strategy='random')

    question_id = item['id']
    difficulty = item['difficulty']
    source = item['source']

    # 获取指定数量的 sampled_solutions
    sampled_solutions = item['sampled_solutions'][:num_codes]
    codes = [sol['code'] for sol in sampled_solutions]

    # 解析 input_output
    io_data = json.loads(item['input_output'])
    test_inputs = io_data['inputs']
    test_outputs = io_data['outputs']

    # 对每个 test case 进行投票
    test_results = []
    correct_count = 0

    for test_idx, (test_input, expected_output) in enumerate(zip(test_inputs, test_outputs)):
        # 规范化 expected_output
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
            'expected_output': expected_output_str.strip(),
            'voted_output': voted_output.strip() if voted_output else None,
            'is_correct': is_correct,
            'vote_counts': vote_result['vote_counts'],
            'success_rate': vote_result['success_rate'],
            'total_votes': vote_result['total_votes'],
            'has_tie': vote_result['has_tie'],
        })

    # 计算整体准确率
    total_tests = len(test_results)
    overall_accuracy = correct_count / total_tests if total_tests > 0 else 0.0

    return {
        'question_id': question_id,
        'difficulty': difficulty,
        'source': source,
        'total_test_cases': total_tests,
        'correct_predictions': correct_count,
        'overall_accuracy': overall_accuracy,
        'test_results': test_results,
        'num_codes_used': num_codes
    }


def process_question_wrapper(args):
    """包装函数用于multiprocessing"""
    item, num_codes = args
    return process_question(item, num_codes)


# ==================== 主函数 ====================
def main(num_codes):
    """
    运行实验

    Args:
        num_codes: 使用的solution数量
    """
    # 配置
    input_file = 'TACO-verified_500_sampled.jsonl'
    output_file = f'voting_results_{num_codes}codes.jsonl'
    log_file = f'voting_experiment_{num_codes}codes.log'
    num_workers = min(64, cpu_count())

    def log(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f'[{timestamp}] {message}'
        print(log_message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    log("=" * 70)
    log(f"开始多数投票实验 - 使用 {num_codes} 个 solutions")
    log("=" * 70)
    log(f"并行进程数: {num_workers}")

    # 读取数据
    log(f"读取数据: {input_file}")
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))

    total_items = len(items)
    log(f"待处理问题数: {total_items}")

    # 清空输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        pass

    # 准备参数
    args_list = [(item, num_codes) for item in items]

    # 并行处理
    results = []
    log("开始并行处理...")

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_question_wrapper, args_list),
            total=total_items,
            desc=f"处理({num_codes}codes)",
            ncols=100
        ):
            results.append(result)

            # 每10个保存一次
            if len(results) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for res in results:
                        f.write(json.dumps(res, ensure_ascii=False) + '\n')
                log(f"已处理 {len(results)}/{total_items} 题，已保存")

    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    log("=" * 70)
    log("实验完成！")
    log("=" * 70)
    log(f"处理问题数: {len(results)}")
    log(f"结果文件: {output_file}")
    log("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python3 run_voting_experiment_config.py <num_codes>")
        print("示例: python3 run_voting_experiment_config.py 4")
        sys.exit(1)

    num_codes = int(sys.argv[1])

    # 验证数量
    if num_codes < 1 or num_codes > 20:
        print(f"错误: num_codes 必须在 1-20 之间，当前值: {num_codes}")
        sys.exit(1)

    main(num_codes)
