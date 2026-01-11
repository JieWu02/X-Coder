#!/usr/bin/env python3
"""
代码执行引擎 - 执行 Python 代码并捕获输出
"""
import subprocess
import tempfile
import os
from typing import Dict, Optional


class CodeExecutor:
    """Python 代码执行器"""

    def __init__(self, timeout: int = 5):
        """
        初始化执行器

        Args:
            timeout: 执行超时时间（秒）
        """
        self.timeout = timeout

    def execute(self, code: str, test_input: str) -> Dict:
        """
        执行 Python 代码

        Args:
            code: Python 代码字符串
            test_input: 测试输入（通过 stdin 传入）

        Returns:
            dict: {
                'success': bool,      # 是否执行成功
                'output': str,        # stdout 输出
                'error': str,         # stderr 错误信息
                'returncode': int,    # 返回码
                'timeout': bool       # 是否超时
            }
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            # 执行代码
            result = subprocess.run(
                ['python3', temp_file],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode,
                'timeout': False
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': None,
                'error': f'Timeout after {self.timeout} seconds',
                'returncode': -1,
                'timeout': True
            }

        except Exception as e:
            return {
                'success': False,
                'output': None,
                'error': f'Execution error: {str(e)}',
                'returncode': -1,
                'timeout': False
            }

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file)
            except:
                pass

    def execute_batch(self, codes: list, test_input: str) -> list:
        """
        批量执行多个代码

        Args:
            codes: 代码列表
            test_input: 测试输入

        Returns:
            list: 执行结果列表
        """
        results = []
        for code in codes:
            result = self.execute(code, test_input)
            results.append(result)
        return results


def test_executor():
    """测试执行器"""
    print("=" * 70)
    print("测试代码执行引擎")
    print("=" * 70)

    executor = CodeExecutor(timeout=5)

    # 测试1: 正常执行
    print("\n测试1: 正常执行")
    code1 = """
n = int(input())
print(n * 2)
"""
    result1 = executor.execute(code1, "5\n")
    print(f"输入: 5")
    print(f"输出: {result1['output'].strip()}")
    print(f"成功: {result1['success']}")

    # 测试2: 运行错误
    print("\n测试2: 运行错误")
    code2 = """
n = int(input())
print(n / 0)
"""
    result2 = executor.execute(code2, "5\n")
    print(f"成功: {result2['success']}")
    print(f"错误: {result2['error'][:100]}")

    # 测试3: 超时
    print("\n测试3: 超时（设置1秒超时）")
    executor_short = CodeExecutor(timeout=1)
    code3 = """
import time
time.sleep(3)
print("done")
"""
    result3 = executor_short.execute(code3, "")
    print(f"超时: {result3['timeout']}")
    print(f"错误: {result3['error']}")

    # 测试4: 使用 sys.stdin.read()
    print("\n测试4: sys.stdin.read() 方式")
    code4 = """
import sys
data = sys.stdin.read().strip().split()
n = int(data[0])
a = int(data[1])
b = int(data[2])
print(n + a + b)
"""
    result4 = executor.execute(code4, "10\n20\n30\n")
    print(f"输入: 10 20 30")
    print(f"输出: {result4['output'].strip()}")
    print(f"成功: {result4['success']}")

    print("\n" + "=" * 70)
    print("✓ 代码执行引擎测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_executor()
