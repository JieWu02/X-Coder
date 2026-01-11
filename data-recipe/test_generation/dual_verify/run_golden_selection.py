#!/usr/bin/env python3
"""
Golden Solution 选择实验 - 完全参照 voting 脚本的并行逻辑
"""
import json
import sys
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from code_executor import CodeExecutor
from majority_voting import MajorityVoter


def process_single_question(args):
    """
    处理单个问题（在独立进程中）

    Args:
        args: (voting_result, original_item, num_codes)

    Returns:
        dict: golden solution结果
    """
    voting_result, original_item, num_codes = args

    # 每个进程创建自己的executor
    executor = CodeExecutor(timeout=5)
    voter = MajorityVoter(tie_break_strategy='random')

    question_id = voting_result['question_id']

    # 获取solutions
    sampled_solutions = original_item['sampled_solutions'][:num_codes]
    codes = [sol['code'] for sol in sampled_solutions]

    # 解析input_output
    io_data = json.loads(original_item['input_output'])
    test_inputs = io_data['inputs']
    test_outputs = io_data['outputs']

    # 构建虚拟测试用例
    virtual_tests = []
    for test_result in voting_result['test_results']:
        if test_result['voted_output'] is not None:
            virtual_tests.append({
                'input': test_inputs[test_result['test_index']],
                'expected': test_result['voted_output']
            })

    # 测试每个solution在虚拟测试上的通过数
    solution_scores = []
    for code_idx, code in enumerate(codes):
        passed = 0
        for vtest in virtual_tests:
            result = executor.execute(code, vtest['input'])
            if result['success'] and result['output'].strip() == vtest['expected'].strip():
                passed += 1
        solution_scores.append({
            'index': code_idx,
            'passed': passed,
            'total': len(virtual_tests)
        })

    # 选择golden solution
    golden = max(solution_scores, key=lambda x: x['passed'])
    golden_code = codes[golden['index']]

    # 测试golden solution在真实测试上
    passed_real = 0
    for test_input, expected_output in zip(test_inputs, test_outputs):
        # 规范化expected_output
        if isinstance(expected_output, list):
            expected_str = json.dumps(expected_output, ensure_ascii=False).strip()
        else:
            expected_str = str(expected_output).strip()

        result = executor.execute(golden_code, test_input)
        if result['success'] and result['output'].strip() == expected_str:
            passed_real += 1

    total_tests = len(test_inputs)

    return {
        'question_id': question_id,
        'difficulty': voting_result['difficulty'],
        'source': voting_result['source'],
        'num_codes': num_codes,
        'golden_index': golden['index'],
        'virtual_passed': golden['passed'],
        'virtual_total': golden['total'],
        'real_passed': passed_real,
        'real_total': total_tests,
        'real_pass_rate': passed_real / total_tests if total_tests > 0 else 0.0,
        'all_tests_passed': (passed_real == total_tests)
    }


def main(num_codes):
    """主函数"""
    # 配置
    voting_file = f'voting_results_{num_codes}codes.jsonl' if num_codes != 8 else 'voting_results.jsonl'
    original_file = 'TACO-verified_500_sampled.jsonl'
    output_file = f'golden_solution_results_{num_codes}codes.jsonl'
    log_file = f'golden_selection_{num_codes}codes.log'
    num_workers = min(64, cpu_count())

    def log(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f'[{timestamp}] {message}'
        print(log_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\\n')

    log("=" * 70)
    log(f"Golden Solution 选择实验 - 使用 {num_codes} 个 solutions")
    log("=" * 70)
    log(f"并行进程数: {num_workers}")

    # 加载数据
    log(f"加载投票结果: {voting_file}")
    voting_results = []
    with open(voting_file, 'r', encoding='utf-8') as f:
        for line in f:
            voting_results.append(json.loads(line.strip()))

    log(f"加载原始数据: {original_file}")
    original_data = {}
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            original_data[item['id']] = item

    total_items = len(voting_results)
    log(f"待处理问题数: {total_items}")

    # 准备参数
    args_list = []
    for voting_result in voting_results:
        qid = voting_result['question_id']
        if qid in original_data:
            args_list.append((voting_result, original_data[qid], num_codes))

    # 清空输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        pass

    # 并行处理
    results = []
    log("开始并行处理...")

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_question, args_list),
            total=len(args_list),
            desc=f"处理({num_codes}codes)",
            ncols=100
        ):
            results.append(result)

            # 每10个保存一次
            if len(results) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for res in results:
                        f.write(json.dumps(res, ensure_ascii=False) + '\n')
                log(f"已处理 {len(results)}/{len(args_list)} 题，已保存")

    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    log("=" * 70)
    log("实验完成！")
    log("=" * 70)

    # 统计
    fully_passed = sum(1 for r in results if r['all_tests_passed'])
    avg_pass_rate = sum(r['real_pass_rate'] for r in results) / len(results) if results else 0

    log(f"处理问题数: {len(results)}")
    log(f"平均通过率: {avg_pass_rate * 100:.2f}%")
    log(f"完全通过数: {fully_passed}")
    log(f"完全通过率: {fully_passed / len(results) * 100:.2f}%")
    log(f"结果文件: {output_file}")
    log("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python3 run_golden_selection_experiment.py <num_codes>")
        print("示例: python3 run_golden_selection_experiment.py 8")
        sys.exit(1)

    num_codes = int(sys.argv[1])
    if num_codes not in [4, 8, 16]:
        print(f"错误: num_codes 必须是 4, 8, 或 16")
        sys.exit(1)

    main(num_codes)
