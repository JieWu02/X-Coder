#!/usr/bin/env python3
"""
从 TACO-verified_with_o3_samples.jsonl 中提取 Python 代码块
- 从 sampled_answers 中提取 Python 代码
- 从 solutions 中提取 Python 代码
- 支持多种格式的代码块
"""
import json
import re
from tqdm import tqdm

def extract_python_code(text):
    """
    从文本中提取 Python 代码块
    支持多种格式：
    1. ```python ... ```
    2. ```Python ... ```
    3. ``` ... ```（如果看起来像 Python）
    4. 纯代码（如果整个文本看起来像代码）
    """
    if not text or not isinstance(text, str):
        return None

    # 方法 1: 提取 ```python ... ``` 格式
    pattern1 = r'```[Pp]ython\s*(.*?)```'
    matches = re.findall(pattern1, text, re.DOTALL)
    if matches:
        # 返回最长的代码块
        return max(matches, key=len).strip()

    # 方法 2: 提取 ``` ... ``` 格式（任意语言）
    pattern2 = r'```\s*(.*?)```'
    matches = re.findall(pattern2, text, re.DOTALL)
    if matches:
        # 返回最长的代码块
        code = max(matches, key=len).strip()
        # 检查是否可能是 Python 代码
        if is_likely_python(code):
            return code

    # 方法 3: 如果整个文本看起来像代码，直接返回
    if is_likely_python(text):
        return text.strip()

    return None


def is_likely_python(text):
    """简单判断文本是否可能是 Python 代码"""
    if not text:
        return False

    # Python 关键字
    python_keywords = [
        'def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
        'return', 'print(', 'range(', '__name__', 'elif ', 'else:'
    ]

    # 检查是否包含足够多的 Python 特征
    keyword_count = sum(1 for kw in python_keywords if kw in text)
    return keyword_count >= 2


def extract_all_codes(item):
    """从单个数据项中提取所有代码"""
    result = {
        'id': item.get('id'),
        'question': item.get('question', '')[:200] + '...',  # 截断问题
        'source': item.get('source'),
        'difficulty': item.get('difficulty'),
        'original_solutions': [],
        'sampled_solutions': [],
    }

    # 提取原始 solutions
    if 'solutions' in item and item['solutions']:
        for i, sol in enumerate(item['solutions']):
            code = extract_python_code(sol)
            if code:
                result['original_solutions'].append({
                    'index': i,
                    'code': code,
                    'raw_length': len(sol)
                })

    # 提取 sampled_answers
    if 'sampled_answers' in item and item['sampled_answers']:
        for i, answer in enumerate(item['sampled_answers']):
            # 跳过错误标记
            if isinstance(answer, str) and answer.startswith('[ERROR]'):
                continue

            code = extract_python_code(answer)
            if code:
                result['sampled_solutions'].append({
                    'index': i,
                    'code': code,
                    'raw_length': len(answer)
                })

    return result


def main():
    input_file = "TACO-verified_with_o3_samples.jsonl"
    output_file = "TACO-verified_extracted_codes.jsonl"

    print("=" * 70)
    print("提取 Python 代码块")
    print("=" * 70)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print()

    # 统计信息
    total_items = 0
    items_with_original = 0
    items_with_sampled = 0
    total_original_codes = 0
    total_sampled_codes = 0
    failed_extractions = 0

    # 处理数据
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="处理数据"):
            total_items += 1
            item = json.loads(line.strip())

            # 提取代码
            result = extract_all_codes(item)

            # 统计
            if result['original_solutions']:
                items_with_original += 1
                total_original_codes += len(result['original_solutions'])

            if result['sampled_solutions']:
                items_with_sampled += 1
                total_sampled_codes += len(result['sampled_solutions'])

            if not result['original_solutions'] and not result['sampled_solutions']:
                failed_extractions += 1

            # 保存
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 输出统计
    print()
    print("=" * 70)
    print("提取完成！")
    print("=" * 70)
    print(f"总条目数: {total_items}")
    print()
    print(f"原始 Solutions:")
    print(f"  - 有代码的条目: {items_with_original} ({items_with_original/total_items*100:.1f}%)")
    print(f"  - 提取的代码块总数: {total_original_codes}")
    print(f"  - 平均每条: {total_original_codes/items_with_original:.1f}" if items_with_original > 0 else "  - 平均每条: 0")
    print()
    print(f"采样 Solutions:")
    print(f"  - 有代码的条目: {items_with_sampled} ({items_with_sampled/total_items*100:.1f}%)")
    print(f"  - 提取的代码块总数: {total_sampled_codes}")
    print(f"  - 平均每条: {total_sampled_codes/items_with_sampled:.1f}" if items_with_sampled > 0 else "  - 平均每条: 0")
    print()
    print(f"提取失败（两者都没有）: {failed_extractions}")
    print()
    print(f"✓ 结果已保存到: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
