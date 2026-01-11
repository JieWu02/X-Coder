"""
Question Generation Templates Package

Contains prompt templates for the two-stage competitive programming problem generation process.
"""

from .select_feature import STAGE1_PROMPT_TEMPLATE
from .codeforces_question_gen import STAGE2_PROMPT_TEMPLATE as CODEFORCES_TEMPLATE
from .leetcode_question_gen import STAGE2_PROMPT_TEMPLATE as LEETCODE_TEMPLATE
from .atcoder_question_gen import STAGE2_PROMPT_TEMPLATE as ATCODER_TEMPLATE

__all__ = [
    'STAGE1_PROMPT_TEMPLATE', 
    'CODEFORCES_TEMPLATE', 
    'LEETCODE_TEMPLATE', 
    'ATCODER_TEMPLATE'
]