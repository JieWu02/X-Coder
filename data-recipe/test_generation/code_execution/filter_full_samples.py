#!/usr/bin/env python3
"""
筛选出采样 Solutions 满额（20个）的条目
"""
import json
from tqdm import tqdm

def filter_full_samples():
    input_file = "TACO-verified_extracted_codes.jsonl"
    output_file = "TACO-verified_full_samples_only.jsonl"

    print("=" * 70)
    print("筛选采样 Solutions 满额的条目")
    print("=" * 70)

    total_count = 0
    full_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="筛选数据"):
            total_count += 1
            item = json.loads(line.strip())

            # 检查 sampled_solutions 是否满额（20个）
            if len(item['sampled_solutions']) == 20:
                full_count += 1
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print()
    print("=" * 70)
    print("筛选完成！")
    print("=" * 70)
    print(f"原始条目数: {total_count}")
    print(f"满额条目数: {full_count} ({full_count/total_count*100:.1f}%)")
    print(f"过滤条目数: {total_count - full_count} ({(total_count - full_count)/total_count*100:.1f}%)")
    print()
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    filter_full_samples()
