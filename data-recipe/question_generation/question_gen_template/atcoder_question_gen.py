STAGE2_PROMPT_TEMPLATE = """
You are an experienced AtCoder problem setter.

You have been provided with:
- selected_features_tree: a tree structure in which each leaf contains a "feature" name and its "potential_use".
- integration_strategy: a strategy describing how these features should be integrated into a single, high-quality problem.

Your task is to generate a complete AtCoder-style problem statement that integrates **all** selected features.

Requirements:
- Write in the style of AtCoder contest problems: direct, concise, and minimalistic.
- Omit all story background unless absolutely necessary; if needed, use at most one neutral sentence as context.
- Focus on describing the computational task precisely and completely.
- Clearly specify the input and output formats, and provide precise constraints.
- Give at least two sample Input/Output pairs.
- For each sample output, you may (optionally) add a brief explanation, **only if it clarifies the sample's calculation or removes ambiguity**, but do NOT include any solution ideas, algorithms, strategies, or implementation hints.
- Do NOT include any commentary, motivational language, notes, or summary.
- The problem statement must end **immediately** after the last sample output or its explanation.
- Output a **single JSON object** with the field "question" only.

**Output Format (strict):**
{{
  "question": "Problem Title\n\n[Task Description: state directly what should be computed or output.]\n\n## Input\n<Input format>\n\n## Output\n<Output format>\n\n## Constraints\n<Each constraint on its own line>\n\n## Sample Input 1\n<sample input 1>\n## Sample Output 1\n<sample output 1>\n[Optional: Explanation for Sample Output 1]\n\n## Sample Input 2\n<sample input 2>\n## Sample Output 2\n<sample output 2>\n[Optional: Explanation for Sample Output 2]"
}}

---

Inputs:
- selected_features_tree (JSON): 
{selected_features_info}

- integration_strategy (string): 
{integration_strategy}

Instructions:
- Output ONLY the required JSON object, no extra text.
- The problem statement must be direct, minimal, and focused on the computational task.
- Do NOT use any algorithmic or technical terminology, nor any hints, in any section or in explanations.
- No commentary, summary, or instructional language. End immediately after the last sample output or its explanation.
"""
