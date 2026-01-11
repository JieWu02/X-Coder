#!/usr/bin/env python3
"""
过滤 test case，每个问题保留最多 20 个 test case
- 只处理 full_samples_only 中的 1,187 个问题
- 限制每个问题的 input_output 为最多 20 个 test case
"""
import json
from tqdm import tqdm

def filter_testcases():
    # 输入输出文件
    full_samples_file = "TACO-verified_full_samples_only.jsonl"
    original_samples_file = "TACO-verified_with_o3_samples.jsonl"
    output_file = "TACO-verified_full_samples_20tc.jsonl"

    print("=" * 70)
    print("过滤 Test Case 到 20 个")
    print("=" * 70)

    # 读取 full_samples 中的 ID 列表
    print(f"\n1. 读取筛选后的问题 ID...")
    valid_ids = set()
    with open(full_samples_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            valid_ids.add(item['id'])

    print(f"   ✓ 需要处理的问题数: {len(valid_ids)}")

    # 读取原始采样文件，过滤 test case
    print(f"\n2. 从原始文件中读取并过滤 test case...")

    total_count = 0
    filtered_count = 0
    original_tc_total = 0
    filtered_tc_total = 0

    with open(original_samples_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="处理数据", total=1300):
            item = json.loads(line.strip())

            # 只处理在 valid_ids 中的问题
            if item['id'] not in valid_ids:
                continue

            total_count += 1

            # 解析 input_output
            try:
                io_data = json.loads(item['input_output'])
            except:
                print(f"   ⚠ 警告: 问题 {item['id']} 的 input_output 解析失败")
                continue

            # 统计原始 test case 数量
            original_tc_count = len(io_data.get('inputs', []))
            original_tc_total += original_tc_count

            # 过滤到最多 20 个 test case
            if original_tc_count > 20:
                io_data['inputs'] = io_data['inputs'][:20]
                io_data['outputs'] = io_data['outputs'][:20]
                filtered_count += 1

            filtered_tc_count = len(io_data['inputs'])
            filtered_tc_total += filtered_tc_count

            # 更新 item 的 input_output
            item['input_output'] = json.dumps(io_data)

            # 写入输出文件
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 统计信息
    print()
    print("=" * 70)
    print("过滤完成！")
    print("=" * 70)
    print(f"处理的问题数: {total_count}")
    print(f"需要过滤的问题数: {filtered_count} ({filtered_count/total_count*100:.1f}%)")
    print(f"未修改的问题数: {total_count - filtered_count} ({(total_count - filtered_count)/total_count*100:.1f}%)")
    print()
    print(f"原始 Test Case 总数: {original_tc_total:,}")
    print(f"过滤后 Test Case 总数: {filtered_tc_total:,}")
    print(f"平均每题 Test Case (原始): {original_tc_total/total_count:.1f}")
    print(f"平均每题 Test Case (过滤后): {filtered_tc_total/total_count:.1f}")
    print()
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    filter_testcases()
