#!/usr/bin/env python3
"""
筛选 testcase 数量 >= 8 的条目
"""
import json
from tqdm import tqdm

def filter_dataset(input_file, output_file, min_testcases=8):
    print("=" * 60)
    print(f"筛选 testcase >= {min_testcases} 的条目...")
    print("=" * 60)

    total_count = 0
    filtered_count = 0
    testcase_stats = []

    # 读取并筛选数据
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="筛选数据"):
            total_count += 1
            try:
                item = json.loads(line.strip())

                # 解析 input_output 字段
                if 'input_output' in item and item['input_output']:
                    io_data = json.loads(item['input_output'])

                    # 计算 testcase 数量
                    num_testcases = 0
                    if 'inputs' in io_data:
                        num_testcases = len(io_data['inputs'])

                    testcase_stats.append(num_testcases)

                    # 筛选条件：testcase >= min_testcases
                    if num_testcases >= min_testcases:
                        filtered_count += 1
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    # 没有 input_output 字段的跳过
                    testcase_stats.append(0)

            except Exception as e:
                print(f"\n警告：处理第 {total_count} 行时出错: {e}")
                continue

    # 输出统计信息
    print(f"\n筛选完成！")
    print("=" * 60)
    print(f"统计信息:")
    print(f"- 原始条目数: {total_count}")
    print(f"- 筛选后条目数: {filtered_count}")
    print(f"- 筛选比例: {filtered_count/total_count*100:.2f}%")
    print(f"\nTestcase 数量分布:")
    print(f"- 最小值: {min(testcase_stats)}")
    print(f"- 最大值: {max(testcase_stats)}")
    print(f"- 平均值: {sum(testcase_stats)/len(testcase_stats):.2f}")
    print(f"- >= {min_testcases} 的条目数: {filtered_count}")
    print("=" * 60)
    print(f"\n✓ 筛选结果已保存到: {output_file}")

    return filtered_count

if __name__ == "__main__":
    input_file = "TACO-verified_original.jsonl"
    output_file = "TACO-verified_filtered_testcase_ge_8.jsonl"

    filtered_count = filter_dataset(input_file, output_file, min_testcases=8)
    print(f"\n下一步：为 {filtered_count} 个条目的 question 进行 16 次 API 采样")
