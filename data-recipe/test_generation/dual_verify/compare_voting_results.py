#!/usr/bin/env python3
"""
多数投票结果对比分析
对比 4、8、16 个solution的投票结果
"""
import json
from collections import defaultdict, Counter


def load_results(filename):
    """加载结果文件"""
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def analyze_single_config(results, num_codes):
    """分析单个配置的结果"""
    stats = {
        'num_codes': num_codes,
        'total_questions': len(results),
        'total_test_cases': 0,
        'total_correct': 0,
        'overall_accuracy': 0.0,

        # 按难度统计
        'by_difficulty': defaultdict(lambda: {'total': 0, 'correct': 0}),

        # 按来源统计
        'by_source': defaultdict(lambda: {'total': 0, 'correct': 0}),

        # 投票一致性分布
        'voting_consistency': Counter(),

        # 成功率统计
        'avg_success_rate': 0.0,
        'success_rates': [],

        # Tie 统计
        'total_ties': 0,
        'tie_rate': 0.0,
    }

    for result in results:
        difficulty = result['difficulty']
        source = result['source']
        test_cases = result['total_test_cases']
        correct = result['correct_predictions']

        # 整体统计
        stats['total_test_cases'] += test_cases
        stats['total_correct'] += correct

        # 按难度
        stats['by_difficulty'][difficulty]['total'] += test_cases
        stats['by_difficulty'][difficulty]['correct'] += correct

        # 按来源
        stats['by_source'][source]['total'] += test_cases
        stats['by_source'][source]['correct'] += correct

        # 投票一致性和成功率
        for test in result['test_results']:
            # 成功率
            stats['success_rates'].append(test['success_rate'])

            # Tie
            if test['has_tie']:
                stats['total_ties'] += 1

            # 投票一致性：计算最高票数占总票数的比例
            if test['vote_counts']:
                max_votes = max(test['vote_counts'].values())
                total_votes = test['total_votes']
                if total_votes > 0:
                    consistency = max_votes / total_votes
                    stats['voting_consistency'][consistency] += 1

    # 计算平均值
    if stats['total_test_cases'] > 0:
        stats['overall_accuracy'] = stats['total_correct'] / stats['total_test_cases']
        stats['tie_rate'] = stats['total_ties'] / stats['total_test_cases']

    if stats['success_rates']:
        stats['avg_success_rate'] = sum(stats['success_rates']) / len(stats['success_rates'])

    # 计算各维度准确率
    for difficulty in stats['by_difficulty']:
        total = stats['by_difficulty'][difficulty]['total']
        correct = stats['by_difficulty'][difficulty]['correct']
        stats['by_difficulty'][difficulty]['accuracy'] = correct / total if total > 0 else 0.0

    for source in stats['by_source']:
        total = stats['by_source'][source]['total']
        correct = stats['by_source'][source]['correct']
        stats['by_source'][source]['accuracy'] = correct / total if total > 0 else 0.0

    return stats


def generate_comparison_report(stats_list):
    """生成对比报告"""
    report = []

    report.append("=" * 80)
    report.append("多数投票结果对比分析报告")
    report.append("=" * 80)
    report.append("")

    # 1. 整体准确率对比
    report.append("=" * 80)
    report.append("1. 整体准确率对比")
    report.append("=" * 80)
    report.append("")
    report.append(f"{'配置':<15} {'总测试数':<12} {'正确数':<12} {'准确率':<12}")
    report.append("-" * 80)

    for stats in stats_list:
        num_codes = stats['num_codes']
        total = stats['total_test_cases']
        correct = stats['total_correct']
        accuracy = stats['overall_accuracy'] * 100
        report.append(f"{num_codes}个solution{'':<6} {total:<12} {correct:<12} {accuracy:.2f}%")

    report.append("")

    # 计算准确率提升
    if len(stats_list) >= 2:
        report.append("准确率变化趋势:")
        for i in range(1, len(stats_list)):
            prev_acc = stats_list[i-1]['overall_accuracy'] * 100
            curr_acc = stats_list[i]['overall_accuracy'] * 100
            diff = curr_acc - prev_acc
            prev_codes = stats_list[i-1]['num_codes']
            curr_codes = stats_list[i]['num_codes']
            report.append(f"  {prev_codes}→{curr_codes}个solution: {diff:+.2f}% "
                         f"({'提升' if diff > 0 else '下降' if diff < 0 else '不变'})")

    report.append("")
    report.append("")

    # 2. 代码执行成功率对比
    report.append("=" * 80)
    report.append("2. 代码执行成功率对比")
    report.append("=" * 80)
    report.append("")
    report.append(f"{'配置':<15} {'平均成功率':<15}")
    report.append("-" * 80)

    for stats in stats_list:
        num_codes = stats['num_codes']
        success_rate = stats['avg_success_rate'] * 100
        report.append(f"{num_codes}个solution{'':<6} {success_rate:.2f}%")

    report.append("")
    report.append("")

    # 3. 投票Tie率对比
    report.append("=" * 80)
    report.append("3. 投票Tie率对比")
    report.append("=" * 80)
    report.append("")
    report.append(f"{'配置':<15} {'Tie次数':<12} {'Tie率':<12}")
    report.append("-" * 80)

    for stats in stats_list:
        num_codes = stats['num_codes']
        ties = stats['total_ties']
        tie_rate = stats['tie_rate'] * 100
        report.append(f"{num_codes}个solution{'':<6} {ties:<12} {tie_rate:.2f}%")

    report.append("")
    report.append("")

    # 4. 按难度对比
    report.append("=" * 80)
    report.append("4. 按难度对比")
    report.append("=" * 80)
    report.append("")

    # 获取所有难度
    all_difficulties = set()
    for stats in stats_list:
        all_difficulties.update(stats['by_difficulty'].keys())

    difficulty_order = ['EASY', 'MEDIUM', 'MEDIUM_HARD', 'HARD', 'VERY_HARD']
    all_difficulties = [d for d in difficulty_order if d in all_difficulties]

    for difficulty in all_difficulties:
        report.append(f"{difficulty}:")
        report.append(f"  {'配置':<15} {'测试数':<10} {'正确数':<10} {'准确率':<10}")
        report.append("  " + "-" * 50)

        for stats in stats_list:
            if difficulty in stats['by_difficulty']:
                num_codes = stats['num_codes']
                total = stats['by_difficulty'][difficulty]['total']
                correct = stats['by_difficulty'][difficulty]['correct']
                accuracy = stats['by_difficulty'][difficulty]['accuracy'] * 100
                report.append(f"  {num_codes}个solution{'':<6} {total:<10} {correct:<10} {accuracy:.2f}%")

        report.append("")

    report.append("")

    # 5. 按来源对比
    report.append("=" * 80)
    report.append("5. 按来源对比")
    report.append("=" * 80)
    report.append("")

    # 获取所有来源
    all_sources = set()
    for stats in stats_list:
        all_sources.update(stats['by_source'].keys())

    for source in sorted(all_sources):
        report.append(f"{source}:")
        report.append(f"  {'配置':<15} {'测试数':<10} {'正确数':<10} {'准确率':<10}")
        report.append("  " + "-" * 50)

        for stats in stats_list:
            if source in stats['by_source']:
                num_codes = stats['num_codes']
                total = stats['by_source'][source]['total']
                correct = stats['by_source'][source]['correct']
                accuracy = stats['by_source'][source]['accuracy'] * 100
                report.append(f"  {num_codes}个solution{'':<6} {total:<10} {correct:<10} {accuracy:.2f}%")

        report.append("")

    report.append("")

    # 6. 投票一致性分析
    report.append("=" * 80)
    report.append("6. 投票一致性分析")
    report.append("=" * 80)
    report.append("")
    report.append("一致性 = 最高票数 / 总票数")
    report.append("")

    # 定义一致性区间
    consistency_bins = [
        (0.25, 0.5, "低一致性 (0.25-0.5)"),
        (0.5, 0.75, "中等一致性 (0.5-0.75)"),
        (0.75, 0.9, "高一致性 (0.75-0.9)"),
        (0.9, 1.0, "极高一致性 (0.9-1.0)"),
        (1.0, 1.0, "完全一致 (1.0)"),
    ]

    for stats in stats_list:
        num_codes = stats['num_codes']
        report.append(f"{num_codes}个solution:")
        total_tests = sum(stats['voting_consistency'].values())

        for min_val, max_val, label in consistency_bins:
            count = sum(cnt for consistency, cnt in stats['voting_consistency'].items()
                       if (consistency > min_val and consistency <= max_val) or
                          (min_val == max_val and consistency == min_val))
            percentage = (count / total_tests * 100) if total_tests > 0 else 0
            report.append(f"  {label:<25} {count:>6} ({percentage:>5.2f}%)")

        report.append("")

    report.append("")

    # 7. 总结与建议
    report.append("=" * 80)
    report.append("7. 总结与建议")
    report.append("=" * 80)
    report.append("")

    # 找出最佳配置
    best_config = max(stats_list, key=lambda s: s['overall_accuracy'])
    report.append(f"最高准确率配置: {best_config['num_codes']}个solution "
                 f"(准确率: {best_config['overall_accuracy']*100:.2f}%)")

    # 分析准确率增益
    if len(stats_list) >= 2:
        acc_4 = stats_list[0]['overall_accuracy']
        acc_8 = stats_list[1]['overall_accuracy'] if len(stats_list) > 1 else acc_4
        acc_16 = stats_list[2]['overall_accuracy'] if len(stats_list) > 2 else acc_8

        gain_4_8 = (acc_8 - acc_4) * 100
        gain_8_16 = (acc_16 - acc_8) * 100 if len(stats_list) > 2 else 0

        report.append("")
        report.append("准确率增益分析:")
        report.append(f"  4→8个solution: {gain_4_8:+.2f}%")
        if len(stats_list) > 2:
            report.append(f"  8→16个solution: {gain_8_16:+.2f}%")

        report.append("")
        if gain_4_8 > 0 and gain_8_16 > 0:
            if gain_8_16 < gain_4_8:
                report.append("观察: 准确率增益呈边际递减趋势，更多solution带来的提升逐渐减小")
            else:
                report.append("观察: 准确率随solution数量持续稳定提升")
        elif gain_4_8 > 0 and gain_8_16 <= 0:
            report.append("观察: 8个solution后准确率不再提升，可能已达到最优点")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """主函数"""
    print("加载结果文件...")

    # 加载三个配置的结果
    configs = [
        (4, 'voting_results_4codes.jsonl'),
        (8, 'voting_results.jsonl'),
        (16, 'voting_results_16codes.jsonl'),
    ]

    stats_list = []
    for num_codes, filename in configs:
        try:
            results = load_results(filename)
            stats = analyze_single_config(results, num_codes)
            stats_list.append(stats)
            print(f"  已加载 {num_codes}个solution 的结果: {len(results)} 个问题")
        except FileNotFoundError:
            print(f"  警告: {filename} 不存在，跳过")

    if len(stats_list) == 0:
        print("错误: 没有找到任何结果文件")
        return

    print(f"\n共加载 {len(stats_list)} 个配置的结果")
    print("\n生成对比报告...")

    # 生成报告
    report = generate_comparison_report(stats_list)

    # 保存报告
    report_file = 'voting_comparison_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存到: {report_file}")
    print("\n" + "=" * 80)
    print(report)


if __name__ == '__main__':
    main()
