#!/usr/bin/env python3
"""
多数投票逻辑 - 从多个执行结果中投票选出最终输出
"""
from collections import Counter
import random
from typing import List, Optional, Dict, Tuple


class MajorityVoter:
    """多数投票器"""

    def __init__(self, tie_break_strategy: str = 'random'):
        """
        初始化投票器

        Args:
            tie_break_strategy: 平票处理策略
                - 'random': 随机选择
                - 'first': 选择第一个出现的
                - 'lexical': 选择字典序最小的
        """
        self.tie_break_strategy = tie_break_strategy

    def vote(self, execution_results: List[Dict]) -> Dict:
        """
        对执行结果进行多数投票

        Args:
            execution_results: 执行结果列表，每个元素是 execute() 返回的字典

        Returns:
            dict: {
                'voted_output': str,           # 投票得到的输出
                'vote_counts': dict,           # 投票详情 {output: count}
                'success_rate': float,         # 成功执行的比率
                'total_votes': int,            # 总投票数（成功执行的数量）
                'has_tie': bool,              # 是否有平票
                'tie_break_used': bool        # 是否使用了平票处理
            }
        """
        # 过滤出成功执行的结果
        successful_results = [
            r for r in execution_results
            if r.get('success', False) and r.get('output') is not None
        ]

        total_executions = len(execution_results)
        total_votes = len(successful_results)

        # 如果没有成功的执行，返回None
        if total_votes == 0:
            return {
                'voted_output': None,
                'vote_counts': {},
                'success_rate': 0.0,
                'total_votes': 0,
                'has_tie': False,
                'tie_break_used': False
            }

        # 收集所有输出（去除首尾空白进行标准化）
        outputs = [r['output'].strip() for r in successful_results]

        # 统计投票
        vote_counter = Counter(outputs)
        vote_counts = dict(vote_counter)

        # 找出得票最多的输出
        max_votes = max(vote_counter.values())
        top_candidates = [
            output for output, count in vote_counter.items()
            if count == max_votes
        ]

        # 检查是否有平票
        has_tie = len(top_candidates) > 1
        tie_break_used = has_tie

        # 选择最终输出
        if has_tie:
            voted_output = self._break_tie(top_candidates)
        else:
            voted_output = top_candidates[0]

        return {
            'voted_output': voted_output,
            'vote_counts': vote_counts,
            'success_rate': total_votes / total_executions,
            'total_votes': total_votes,
            'has_tie': has_tie,
            'tie_break_used': tie_break_used
        }

    def _break_tie(self, candidates: List[str]) -> str:
        """
        平票处理

        Args:
            candidates: 平票的候选输出列表

        Returns:
            str: 选中的输出
        """
        if self.tie_break_strategy == 'random':
            return random.choice(candidates)
        elif self.tie_break_strategy == 'first':
            return candidates[0]
        elif self.tie_break_strategy == 'lexical':
            return min(candidates)
        else:
            return random.choice(candidates)

    def compare_with_expected(
        self,
        voted_output: Optional[str],
        expected_output: str
    ) -> bool:
        """
        比较投票输出和期望输出

        Args:
            voted_output: 投票得到的输出
            expected_output: 期望输出

        Returns:
            bool: 是否匹配
        """
        if voted_output is None:
            return False

        # 精确字符串匹配（已经 strip 过）
        return voted_output.strip() == expected_output.strip()


def test_majority_voter():
    """测试多数投票器"""
    print("=" * 70)
    print("测试多数投票器")
    print("=" * 70)

    voter = MajorityVoter(tie_break_strategy='random')

    # 测试1: 明显多数
    print("\n测试1: 明显多数（5个相同，3个不同）")
    results1 = [
        {'success': True, 'output': '42\n'},
        {'success': True, 'output': '42\n'},
        {'success': True, 'output': '42\n'},
        {'success': True, 'output': '42\n'},
        {'success': True, 'output': '42\n'},
        {'success': True, 'output': '43\n'},
        {'success': True, 'output': '44\n'},
        {'success': True, 'output': '45\n'},
    ]
    vote_result1 = voter.vote(results1)
    print(f"投票结果: {vote_result1['voted_output'].strip()}")
    print(f"投票详情: {vote_result1['vote_counts']}")
    print(f"成功率: {vote_result1['success_rate']:.1%}")
    print(f"有平票: {vote_result1['has_tie']}")

    # 测试2: 有失败的执行
    print("\n测试2: 有失败的执行（3个失败，5个成功相同）")
    results2 = [
        {'success': True, 'output': '100\n'},
        {'success': True, 'output': '100\n'},
        {'success': True, 'output': '100\n'},
        {'success': True, 'output': '100\n'},
        {'success': True, 'output': '100\n'},
        {'success': False, 'output': None},
        {'success': False, 'output': None},
        {'success': False, 'output': None},
    ]
    vote_result2 = voter.vote(results2)
    print(f"投票结果: {vote_result2['voted_output'].strip()}")
    print(f"总投票数: {vote_result2['total_votes']}/8")
    print(f"成功率: {vote_result2['success_rate']:.1%}")

    # 测试3: 平票情况
    print("\n测试3: 平票情况（4:4）")
    results3 = [
        {'success': True, 'output': 'YES\n'},
        {'success': True, 'output': 'YES\n'},
        {'success': True, 'output': 'YES\n'},
        {'success': True, 'output': 'YES\n'},
        {'success': True, 'output': 'NO\n'},
        {'success': True, 'output': 'NO\n'},
        {'success': True, 'output': 'NO\n'},
        {'success': True, 'output': 'NO\n'},
    ]
    vote_result3 = voter.vote(results3)
    print(f"投票结果: {vote_result3['voted_output'].strip()}")
    print(f"投票详情: {vote_result3['vote_counts']}")
    print(f"有平票: {vote_result3['has_tie']}")
    print(f"使用平票处理: {vote_result3['tie_break_used']}")

    # 测试4: 全部失败
    print("\n测试4: 全部失败")
    results4 = [
        {'success': False, 'output': None},
        {'success': False, 'output': None},
        {'success': False, 'output': None},
        {'success': False, 'output': None},
        {'success': False, 'output': None},
        {'success': False, 'output': None},
        {'success': False, 'output': None},
        {'success': False, 'output': None},
    ]
    vote_result4 = voter.vote(results4)
    print(f"投票结果: {vote_result4['voted_output']}")
    print(f"总投票数: {vote_result4['total_votes']}/8")

    # 测试5: 与期望输出比较
    print("\n测试5: 与期望输出比较")
    expected = "42"
    is_correct1 = voter.compare_with_expected("42\n", expected)
    is_correct2 = voter.compare_with_expected("43\n", expected)
    is_correct3 = voter.compare_with_expected(None, expected)
    print(f"'42\\n' == '42': {is_correct1}")
    print(f"'43\\n' == '42': {is_correct2}")
    print(f"None == '42': {is_correct3}")

    print("\n" + "=" * 70)
    print("✓ 多数投票器测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_majority_voter()
