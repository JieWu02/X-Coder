"""
Stage 2 Prompt Template for Codeforces Question Generation
"""

STAGE2_PROMPT_TEMPLATE = """You are a professional competitive programming problem setter.

You have been provided with:

- selected_features_tree: a tree structure where each leaf contains a "feature" name and its "potential_use".
- integration_strategy: a strategy describing how these features should be integrated into a single, high-quality problem.

Your task is to **generate a complete Codeforces-style problem statement** that fully integrates ALL selected features.

Requirements:
- The story and setting must naturally motivate every selected feature, making each indispensable for an optimal solution.
- Specify precise input/output format and tight constraints.
- Provide at least two distinct, non-trivial sample Input/Output pairs, each with a clear explanation.
- Make sure the samples are consistent with your constraints and the solution requires use of all selected features.
- Do not include any references to algorithms, data structures, solution strategies, or any implicit or explicit hints in any part of the statement, notes, or examples. Do not include any motivational, summary, or instructional phrases (e.g., "Remember", etc.) at any point in the output. The statement must end after the final example or clarification, with no extraneous commentary.
- Output should be a **single JSON object** with the field "question" only.

**Output Format (strictly):**

{{
  "question": "# Problem Title\\n\\nStory/context (describe the scenario)\\n\\n## Input\\n<...input description...>\\n\\n## Output\\n<...output description...>\\n\\n## Example\\n### Input\\n<code block with sample input>\\n### Output\\n<code block with sample output>\\n### Note\\nExplanation about the sample(s), but without any solution hints."
}}

---

**Inputs:**
- selected_features_tree (JSON): 
{selected_features_info}

- integration_strategy (string): 
{integration_strategy}

---
Instructions:
- You must ensure every selected feature is essential and naturally integrated.
- Output ONLY the required JSON object, no extra text.
"""