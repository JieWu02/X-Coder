#!/usr/bin/env python3
"""
下载 TACO-verified 数据集并保存为 jsonl 格式
"""
import json
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
import pandas as pd

def convert_to_serializable(obj):
    """将不可序列化的对象转换为可序列化格式"""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def download_and_save_dataset():
    print("=" * 60)
    print("开始下载 TACO-verified 数据集...")
    print("=" * 60)

    try:
        # 下载数据集
        dataset = load_dataset("likaixin/TACO-verified")

        # 获取训练集（通常数据集在 'train' split）
        # 如果有其他 split，会自动检测
        if 'train' in dataset:
            data = dataset['train']
        elif 'test' in dataset:
            data = dataset['test']
        else:
            # 如果没有明确的 split，获取第一个可用的
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"使用数据集 split: {split_name}")

        print(f"\n数据集信息:")
        print(f"- 总条目数: {len(data)}")
        print(f"- 字段: {data.column_names}")

        # 保存为 jsonl 格式
        output_file = "TACO-verified_original.jsonl"
        print(f"\n保存数据集到: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(data, desc="写入 jsonl"):
                # 转换为字典并处理不可序列化的类型
                item_dict = convert_to_serializable(dict(item))
                json_line = json.dumps(item_dict, ensure_ascii=False)
                f.write(json_line + '\n')

        print(f"\n✓ 数据集下载完成！")
        print(f"✓ 已保存到: {output_file}")
        print(f"✓ 总条目数: {len(data)}")

        # 显示第一条数据的结构（用于调试）
        print(f"\n示例数据结构（第一条）:")
        print(json.dumps(data[0], ensure_ascii=False, indent=2)[:500] + "...")

        return output_file, len(data)

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        raise

if __name__ == "__main__":
    download_and_save_dataset()
