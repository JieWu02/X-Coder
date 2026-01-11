"""
Stage 1 Prompt Template for Feature Selection
"""

STAGE1_PROMPT_TEMPLATE = """You are a professional competitive programming problem setter.

---

Your task consists of three parts:

Step 1: Tree-Structured Feature Role Explanation

Recursively traverse the provided feature tree.
- For each leaf node, annotate it with a "potential_use" field describing how this feature is typically used in competitive programming problems (e.g., input modeling, optimization, search, handling edge cases, etc.).
- Internal nodes retain their structure for hierarchy.

Output the annotated tree in the same structure, with every leaf node containing its "potential_use".

---

Step 2: Subtree Selection for Problem Integration

Based on your role analysis, select a subtree (tree-structured subset) where all selected leaf features can be naturally integrated into a single, high-quality competitive programming problem.

- Only include features that contribute meaningfully to the same problem idea.
- Internal nodes are included only if they have selected children.
- For each selected leaf, include only its "feature" name and "potential_use".

---

Step 3: Integration Strategy

Briefly describe ("integration_strategy") how the selected features can be integrated together in a single problem, focusing on how their combination enables a meaningful and challenging algorithmic scenario.

---

**Output Format (strictly):**

Return a JSON object **with exactly this structure** (here is an example):

{{
  "feature_roles_tree": {{
    "algorithm": {{
      "search algorithm": {{
        "binary search": {{
          "recursive binary search": {{
            "potential_use": "Used for divide-and-conquer searching in sorted structures or answer spaces."
          }},
          "iterative binary search": {{
            "potential_use": "Efficient loop-based implementation for finding bounds or specific elements."
          }}
        }},
        "breadth-first search (BFS)": {{
          "level-order BFS": {{
            "potential_use": "Traverses graphs layer by layer; useful for shortest path or component discovery."
          }}
        }}
      }}
    }},
    "data structures": {{
      "bitmap": {{
        "bit manipulation": {{
          "bitwise AND": {{
            "potential_use": "Filters or checks properties using bitmasks."
          }},
          "bitwise OR": {{
            "potential_use": "Combines flags or sets with bitwise aggregation."
          }}
        }}
      }}
    }}
  }},

  "selected_features_tree": {{
    "algorithm": {{
      "search algorithm": {{
        "binary search": {{
          "recursive binary search": {{
            "feature": "recursive binary search",
            "potential_use": "Used for divide-and-conquer searching in sorted structures or answer spaces."
          }}
        }}
      }}
    }},
    "data structures": {{
      "bitmap": {{
        "bit manipulation": {{
          "bitwise AND": {{
            "feature": "bitwise AND",
            "potential_use": "Filters or checks properties using bitmasks."
          }}
        }}
      }}
    }}
  }},

  "integration_strategy": "The problem will require recursive binary search to efficiently search over a sorted value space, while bitwise AND operations will be used to filter candidate solutions according to constraints. Their combination allows for a problem that involves searching over sets and optimizing bitwise criteria."
}}

---

**Available Features (Tree):**
{features_json}

---

Instructions:
- Always preserve the tree structure in both "feature_roles_tree" and "selected_features_tree".
- Do not flatten or convert to arrays.
- In selected_features_tree, only include \"feature\" and \"potential_use\" fields for leaf nodes.
- \"integration_strategy\" should make clear how/why these features form a coherent, advanced problem.
- Do not be overly conservative; it is often possible to design advanced problems where many features interact in non-trivial ways. Challenge yourself to maximize feature use without sacrificing problem quality.
"""