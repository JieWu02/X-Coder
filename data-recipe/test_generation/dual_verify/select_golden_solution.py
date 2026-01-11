#!/usr/bin/env python3
"""
Golden Solution 选择实验 - 并行版本

使用 voted outputs 作为虚拟测试用例，选择通过最多的 solution 作为 golden solution，
然后测试 golden solution 在真实 ground truth 上的表现
"""
import json
import sys
from multiprocessing import Pool, cpu_count
from code_executor import CodeExecutor


def load_voting_results(filename):
    """加载投票结果"""
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def load_original_data(filename):
    """加载原始数据"""
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data[item['id']] = item
    return data


def select_golden_solution_single(args):
    """
    处理单个问题的包装函数（用于并行处理）

    Args:
        args: (voting_result, original_item, num_codes)

    Returns:
        dict: golden solution 信息及其在真实测试上的表现
    """
    voting_result, original_item, num_codes = args

    # 在每个进程内创建执行器
    executor = CodeExecutor(timeout=5)

    question_id = voting_result['question_id']

    # 获取前 num_codes 个 solutions
    sampled_solutions = original_item['sampled_solutions'][:num_codes]
    codes = [sol['code'] for sol in sampled_solutions]

    # 解析真实的 input_output
    io_data = json.loads(original_item['input_output'])
    test_inputs = io_data['inputs']
    test_outputs = io_data['outputs']

    # 构建虚拟测试用例（使用 voted outputs）
    virtual_test_cases = []
    for test_result in voting_result['test_results']:
        test_idx = test_result['test_index']
        voted_output = test_result['voted_output']

        if voted_output is not None:  # 只使用有有效投票结果的测试
            virtual_test_cases.append({
                'input': test_inputs[test_idx],
                'expected_output': voted_output
            })

    # 对每个 solution，测试在虚拟测试用例上的通过数
    solution_scores = []
    for code_idx, code in enumerate(codes):
        passed_virtual = 0

        for virtual_test in virtual_test_cases:
            exec_result = executor.execute(code, virtual_test['input'])
            if exec_result['success']:
                actual_output = exec_result['output'].strip()
                expected_output = virtual_test['expected_output'].strip()
                if actual_output == expected_output:
                    passed_virtual += 1

        solution_scores.append({
            'code_index': code_idx,
            'passed_virtual': passed_virtual,
            'total_virtual': len(virtual_test_cases)
        })

    # 选择通过虚拟测试最多的 solution 作为 golden solution
    # 如果有多个solution通过数相同，选择第一个
    golden_solution = max(solution_scores, key=lambda x: x['passed_virtual'])
    golden_code_idx = golden_solution['code_index']
    golden_code = codes[golden_code_idx]

    # 测试 golden solution 在真实 ground truth 上的表现
    passed_real = 0
    real_test_results = []

    for test_idx, (test_input, expected_output) in enumerate(zip(test_inputs, test_outputs)):
        # 规范化 expected_output
        if isinstance(expected_output, list):
            expected_output_str = json.dumps(expected_output, ensure_ascii=False)
        else:
            expected_output_str = str(expected_output)

        # 执行 golden solution
        exec_result = executor.execute(golden_code, test_input)

        is_correct = False
        if exec_result['success']:
            actual_output = exec_result['output'].strip()
            expected_output_normalized = expected_output_str.strip()
            is_correct = (actual_output == expected_output_normalized)
            if is_correct:
                passed_real += 1

        real_test_results.append({
            'test_index': test_idx,
            'success': exec_result['success'],
            'is_correct': is_correct
        })

    # 计算统计信息
    total_real_tests = len(test_inputs)
    pass_rate = passed_real / total_real_tests if total_real_tests > 0 else 0.0
    all_passed = (passed_real == total_real_tests)

    return {
        'question_id': question_id,
        'difficulty': voting_result['difficulty'],
        'source': voting_result['source'],
        'num_codes': num_codes,

        # Golden solution 信息
        'golden_code_index': golden_code_idx,
        'virtual_passed': golden_solution['passed_virtual'],
        'virtual_total': golden_solution['total_virtual'],
        'virtual_pass_rate': golden_solution['passed_virtual'] / golden_solution['total_virtual']
                            if golden_solution['total_virtual'] > 0 else 0.0,

        # 在真实测试上的表现
        'real_passed': passed_real,
        'real_total': total_real_tests,
        'real_pass_rate': pass_rate,
        'all_tests_passed': all_passed,

        # 所有 solutions 在虚拟测试上的得分
        'all_solution_scores': solution_scores
    }


def main(num_codes):
    """
    主函数 - 并行版本

    Args:
        num_codes: 使用的solution数量 (4, 8, or 16)
    """
    # 配置
    voting_result_file = f'voting_results_{num_codes}codes.jsonl' if num_codes != 8 else 'voting_results.jsonl'
    original_data_file = 'TACO-verified_500_sampled.jsonl'
    output_file = f'golden_solution_results_{num_codes}codes.jsonl'
    num_workers = min(64, cpu_count())

    print(f"=" * 70)
    print(f"Golden Solution 选择实验（并行版本）- 使用 {num_codes} 个 solutions")
    print(f"=" * 70)
    print(f"并行进程数: {num_workers}")

    # 加载数据
    print(f"加载投票结果: {voting_result_file}")
    voting_results = load_voting_results(voting_result_file)

    print(f"加载原始数据: {original_data_file}")
    original_data = load_original_data(original_data_file)

    print(f"待处理问题数: {len(voting_results)}")

    # 准备参数
    args_list = []
    for voting_result in voting_results:
        question_id = voting_result['question_id']
        if question_id in original_data:
            original_item = original_data[question_id]
            args_list.append((voting_result, original_item, num_codes))
        else:
            print(f"警告: 问题 {question_id} 在原始数据中不存在")

    # 清空输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        pass

    # 并行处理
    results = []
    print(f"开始并行处理...")

    with Pool(processes=num_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(select_golden_solution_single, args_list)):
            results.append(result)

            # 每10个保存一次并打印进度
            if len(results) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for res in results:
                        f.write(json.dumps(res, ensure_ascii=False) + '\\n')
                print(f"已处理 {len(results)}/{len(args_list)} 题，已保存")

    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\\n')

    print(f"\\n结果已保存到: {output_file}")

    # 计算统计信息
    print(f"\\n" + "=" * 70)
    print("统计结果")
    print("=" * 70)

    total_questions = len(results)
    fully_passed_count = sum(1 for r in results if r['all_tests_passed'])
    avg_pass_rate = sum(r['real_pass_rate'] for r in results) / total_questions if total_questions > 0 else 0.0

    print(f"总问题数: {total_questions}")
    print(f"平均通过率 (Average Pass Rate): {avg_pass_rate * 100:.2f}%")
    print(f"完全通过数 (Fully Passed): {fully_passed_count}")
    print(f"完全通过率 (Full Pass Rate): {fully_passed_count / total_questions * 100:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python3 select_golden_solution_parallel.py <num_codes>")
        print("示例: python3 select_golden_solution_parallel.py 8")
        sys.exit(1)

    num_codes = int(sys.argv[1])

    if num_codes not in [4, 8, 16]:
        print(f"错误: num_codes 必须是 4, 8, 或 16，当前值: {num_codes}")
        sys.exit(1)

    main(num_codes)
