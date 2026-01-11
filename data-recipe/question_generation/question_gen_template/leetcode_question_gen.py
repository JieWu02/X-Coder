"""
Stage 2 Prompt Template for LeetCode Question Generation
"""

STAGE2_PROMPT_TEMPLATE = """You are an expert LeetCode problem setter.

You have been provided with:
- selected_features_tree: a tree structure in which each leaf contains a 'feature' name and its 'potential_use'.
- integration_strategy: a strategy describing how these features should be integrated into a single, high-quality problem.

Your task is to generate a complete LeetCode-style problem statement that integrates **all** selected features.

Requirements:
- Write in the style of official LeetCode problems: concise, neutral, and approachable for a global audience.
- Focus on a single, well-defined computational task. If the features or integration strategy would lead to more than one distinct computational goal, select the most central/main goal and subsume others if possible. Do **not** combine unrelated subtasks.
- Start with a clear, direct statement of the problem. If context is needed, use at most one brief, neutral sentence.
- Do **not** use any references to algorithms, data structures, implementation techniques, hints, or time/space complexity—directly or indirectly. Never mention how to solve the problem, or suggest steps, approaches, or possibilities.
- **Strictly avoid any algorithm names, technical terms, or implementation concepts such as 'DFS', 'BFS', 'dynamic programming', 'recursion', 'greedy', 'divide and conquer', 'flood fill', or similar, anywhere in the statement, examples, or constraints. Only describe the computational task in plain English.**
- Use a concise, descriptive, noun-phrase English title (e.g., 'Longest Increasing Subsequence'). Do not use technical terms in the title.
- In all examples and constraints, use *exact* LeetCode input/output format: e.g., 'Input: nums = [1,2,3], k = 2', 's = \"abc\"', etc.
- Use numbered Examples with bolded headings ('Example 1:', etc.). Each Example must use LeetCode style: Input: ..., Output: ... (as plain text). No code blocks or markdown.
- Always provide at least two non-trivial examples that cover distinct scenarios and edge cases.
- Always provide a 'Constraints' section listing all parameter bounds and important restrictions, each on its own line, using the same variable names as in the statement.
- Do not include any additional commentary, notes, summary, explanations, or instructional language. The statement must end immediately after the last constraint.
- Output a **single JSON object** with the field 'question' only.

**Output Format (strict):**
{{
  "question": "<Title>\n\n<Direct problem description and minimal context if needed.>\n\nExample 1:\nInput: <sample input 1>\nOutput: <sample output 1>\n\nExample 2:\nInput: <sample input 2>\nOutput: <sample output 2>\n\nConstraints:\n<All constraints, each on its own line>"
}}

---

Inputs:
- selected_features_tree (JSON): 
{selected_features_info}

- integration_strategy (string): 
{integration_strategy}

Instructions:
- Output ONLY the required JSON object, no extra text.
- Ensure the problem description is direct, unambiguous, and easy to understand for a global audience.
- No explanations, hints, algorithm names, or commentary. End the statement immediately after the last constraint."
"""
