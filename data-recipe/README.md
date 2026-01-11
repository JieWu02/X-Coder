# SFT Data Recipe: Question, Answer, and Test Synthesis

## Overview

This repository provides tools for synthesizing training data for supervised fine-tuning (SFT) of code generation models across three stages:

1. **Question Generation** - Generate diverse competitive programming questions based on feature trees
2. **Answer Generation** - Generate detailed code solutions with comprehensive explanations
3. **Test Generation** - Build and verify test cases using dual verification

## Structure

```
data-recipe/
├── question_generation/          # Question generation module
│   ├── scripts/
│   │   └── generate_questions.py # Question generation (API)
│   ├── run_question_generate.sh # Start question generation (API client)
│   ├── features_trees_data/
│   │   └── feature_all.jsonl     # Merged feature dataset (JSONL)
│   ├── questions_example.jsonl  # Example questions (JSONL)
│   └── question_gen_template/     # Codeforces, LeetCode, AtCoder templates
│
├── answer_generation/            # Answer generation module
│   ├── scripts/
│   │   ├── gen_answer_batched.py # Core answer logic (batch API)
│   │   └── gen_answer_concurrent.py # Concurrent fallback (no batch API)
│   ├── run_answer_generation.sh # Start server + batch/concurrent examples
│   └── utils/                    # Postprocessing utilities
│       └── filter_valid_python_ast.py
│       └── run_filter_valid_python_ast.sh
│
├── test_generation/              # Test case generation + dual-verify tools
│   ├── download_dataset.py       # Download TACO-verified dataset
│   ├── filter_dataset.py         # Filter items by testcase count
│   ├── solution_sampling/
│   │   ├── sample_solutions_api.py # Sample answers via OpenAI-compatible API
│   │   └── openai_api_client.py  # OpenAI API client
│   ├── code_execution/
│   │   ├── extract_python_code.py # Extract Python code blocks
│   │   ├── filter_full_samples.py # Keep items with full samples
│   │   └── filter_testcases_to_20.py # Keep 20 testcases per item
│   └── dual_verify/
│       ├── code_executor.py       # Execute code with testcases
│       ├── majority_voting.py     # Vote on execution outputs
│       ├── run_majority_voting.py # Run voting pipeline
│       ├── voting_experiment_config.py # Voting configuration
│       ├── compare_voting_results.py # Compare voting outputs
│       ├── select_golden_solution.py # Pick golden solution
│       └── run_golden_selection.py # Run golden selection
│
├── README.md                     # This file
```

## Configuration

**Question Generation**: Configure via environment variables in `run_question_generate.sh`.

**Answer Generation**: Use CLI flags on `gen_answer_batched.py` or `gen_answer_concurrent.py`.

Or use environment variables:
```bash
export SGLANG_ANSWER_WORKER_IP="localhost"
export SGLANG_ANSWER_WORKER_PORT="30001"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your_api_key"
```

## Usage

### 1) Question Generation

```bash
cd question_generation
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="deepseek-ai/DeepSeek-R1-0528"

./run_question_generate.sh 0 1000 ../output/questions_0_1000.jsonl 64
```

Example question JSONL file: `question_generation/questions_example.jsonl`

Input format (feature JSONL, one per line):
```jsonl
{"features": {"problem type": {"subarray manipulation": {"maximum subarray sum": ["Kadane's algorithm", "Divide and Conquer approach"]}}}, "mandatory_features": [], "idx": 0, "leaf_count": 147}
```

Output format (question JSONL, one per line):
```jsonl
{"config_index": 123, "question": "Problem statement ...", "style": "codeforces", "stage1_analysis": {...}, "stage2_output": {...}}
```

### 2) Answer Generation

Start the SGLang server + run batch/concurrent examples:
```bash
cd answer_generation
./run_answer_generation.sh
```

Or run manually (batch API):
```bash
cd answer_generation/scripts
python gen_answer_batched.py \
    --input ../../output/questions/questions_0_1000.jsonl \
    --output ../../output/answers/answers_0_1000.jsonl \
    --model-name deepseek-ai/DeepSeek-R1-0528 \
    --use-batch-api \
    --worker-ip localhost \
    --worker-port 30001
```

Input format (question JSONL, one per line):
```jsonl
{"question": "Line Segment Connectivity ...", "question_index": 0, "original_index": 0, "style": "atcoder"}
```

Output format (answer JSONL, one per line):
```jsonl
{"question_index": 0, "question_content": "Line Segment Connectivity ...", "generated_answer": "```python\\n...\\n```", "extracted_code": "..."}
```

Concurrent Version:
```bash
python gen_answer_concurrent.py \
    --input-file ../../output/questions/questions_0_1000.jsonl \
    --output-file ../../output/answers/answers_0_1000.jsonl \
    --model-name deepseek-ai/DeepSeek-R1-0528 \
    --worker-ips localhost \
    --worker-port 30001 \
    --concurrency 128
```

### 3) Filter Valid Answers

Filtering requirements:
- Exactly one complete ```python code block
- Code block passes ast.parse
- Optional token range via --min-tokens / --max-tokens

```bash
cd answer_generation/utils

./filter_valid_python_ast.py \
    ../../output/answers/answers_0_1000.jsonl \
    -o ../../output/answers/answers_0_1000_filtered.jsonl \
    --min-tokens 2000 \
    --max-tokens 32768
```

Input/Output format (answer JSONL, one per line):
```jsonl
{"question_index": 123, "question_content": "Problem statement ...", "generated_answer": "```python\\n...\\n```", "extracted_code": "..."}
```

## Test Case Generation (Dual Verification)

This pipeline turns TACO-verified items into verified outputs using multi-solution sampling + execution voting.

### Step 1) Download dataset
Purpose: fetch the raw TACO-verified dataset as JSONL for downstream filtering.

```bash
python test_generation/download_dataset.py
```

Input: none (downloads from HuggingFace inside the script).

Output (JSONL, one per line):
```jsonl
{"id": 1, "question": "...", "solutions": ["..."], "input_output": "{\"inputs\": [...], \"outputs\": [...]}"}
```

### Step 2) Filter by testcase count
Purpose: keep only items with enough testcases (quality control).

```bash
python test_generation/filter_dataset.py
```

Input: JSONL from Step 1.

Output (JSONL, one per line):
```jsonl
{"id": 1, "question": "...", "solutions": ["..."], "input_output": "{\"inputs\": [...], \"outputs\": [...]}"}
```

### Step 3) Sample multiple answers (solution sampling)
Purpose: generate multiple candidate solutions per question via OpenAI-compatible API sampling.

```bash
python test_generation/solution_sampling/sample_solutions_api.py
```

Input: filtered JSONL from Step 2.

Output (JSONL, one per line):
```jsonl
{"id": 1, "question": "...", "input_output": "{\"inputs\": [...], \"outputs\": [...]}",
 "sampled_answers": ["answer1", "answer2", "..."]}
```

### Step 4) Extract Python code
Purpose: extract Python code blocks from original and sampled answers.

```bash
python test_generation/code_execution/extract_python_code.py
```

Input: sampled JSONL from Step 3.

Output (JSONL, one per line):
```jsonl
{"id": 1, "question": "...", "input_output": "{\"inputs\": [...], \"outputs\": [...]}",
 "original_solutions": [{"index": 0, "code": "def solve(): ..."}],
 "sampled_solutions": [{"index": 0, "code": "def solve(): ..."}]}
```

### Step 5) Keep fully sampled items
Purpose: retain items that reached the target number of samples.

```bash
python test_generation/code_execution/filter_full_samples.py
```

Input: extracted JSONL from Step 4.

Output: JSONL with only items that have the full required number of samples.

### Step 6) Optional: Trim to 20 testcases
Purpose: cap testcase count to a fixed size for uniform evaluation.

```bash
python test_generation/code_execution/filter_testcases_to_20.py
```

Input: JSONL from Step 5.

Output: JSONL with at most 20 testcases per item.

### Step 7) Majority voting (dual verify)
Purpose: run solutions on testcases and vote to select the most consistent output.

```bash
python test_generation/dual_verify/run_majority_voting.py
```

Input: JSONL with `sampled_solutions` and `input_output`.

Output (JSONL, one per line):
```jsonl
{"question_id": 1, "voted_output": "...", "vote_counts": {"...": 8}, "success_rate": 0.9}
```

### Step 8) Golden solution selection
Purpose: select the best solution using voting results as virtual testcases.

```bash
python test_generation/dual_verify/run_golden_selection.py
```

Input: voting results + original dataset.

Output (JSONL, one per line):
```jsonl
{"question_id": 1, "golden_index": 3, "virtual_passed": 18, "virtual_total": 20}
```
