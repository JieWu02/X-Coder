# X-Coder RL

X-Coder RL training framework for code generation models using reinforcement learning.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Dataset Description](#dataset-description)
- [Training Recipes](#training-recipes)
- [Citation](#citation)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/JieWu02/X-Coder.git
cd X-Coder

# 2. Start Docker container
sudo docker run -it --rm \
  --gpus all \
  --ipc=host \
  -v $(pwd):/workspace \
  whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3 \
  /bin/bash

# 3. Install dependencies
pip install sandbox_fusion pyext
cd rl-recipe
pip install -e .

# 4. Download training data
cd ..
python download_data.py

# 5. Start training
cd rl-recipe
bash train_scripts/install.sh
bash train_scripts/xcoder-rl-train.sh
```

## Environment Setup

### Option 1: Docker (Recommended)

We recommend using our pre-built Docker image which includes all necessary dependencies:

```bash
sudo docker run -it --rm \
  --gpus all \
  --ipc=host \
  -v $(pwd):/workspace \
  whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3 \
  /bin/bash
```

Inside the container, install additional dependencies:

```bash
pip install sandbox_fusion pyext
cd rl-recipe
pip install -e .
```

### Option 2: Manual Installation

If you prefer manual installation:

```bash
# Install verl and dependencies
cd rl-recipe
pip install -e .

# Install additional packages
pip install sandbox_fusion pyext
```

## Data Preparation

The training data (~17GB total) is hosted on HuggingFace: [IIGroup/X-Coder-RL-40k](https://huggingface.co/datasets/IIGroup/X-Coder-RL-40k)

### Download Data

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download all data (~17GB)
python download_data.py

# Or download only synthetic data (~8.5GB)
python download_data.py --syn-only

# Or download only real data (~8.4GB)
python download_data.py --real-only

# Custom output directory
python download_data.py --output-dir ./data
```

### Data Structure

After downloading, the data will be organized as:

```
rl-recipe/
├── syn_rl_data/
│   └── xcoder_data/
│       └── sorted_by_passrate/
│           ├── part_0000.parquet
│           ├── part_0001.parquet
│           ├── part_0002.parquet
│           ├── part_0003.parquet
│           └── rl_tasks_easy.parquet
└── real_rl_data/
    └── non_sys_prompt/
        ├── codeforces_9763.parquet
        ├── klear_code.parquet
        ├── leetcode_2772.parquet
        ├── taco_13064.parquet
        └── test_wo_prompt.parquet
```

## Training

### Step 1: Install Training Dependencies

```bash
cd rl-recipe
bash train_scripts/install.sh
```

### Step 2: Start Training

```bash
bash train_scripts/xcoder-rl-train.sh
```

For Qwen3 models specifically:

```bash
bash train_scripts/xcoder-rl-train-qwen3.sh
```

## Dataset Description

| Dataset | Description | Size |
|---------|-------------|------|
| **syn_rl_data** | Synthetic RL training data | ~8.5GB |
| **real_rl_data** | Real-world code problems | ~8.4GB |

### Synthetic Data (`syn_rl_data`)

- `part_0000.parquet` - `part_0003.parquet`: Main training data sorted by pass rate
- `rl_tasks_easy.parquet`: Easy tasks for curriculum learning

### Real Data (`real_rl_data`)

- `codeforces_9763.parquet`: 9,763 Codeforces problems
- `leetcode_2772.parquet`: 2,772 LeetCode problems
- `taco_13064.parquet`: 13,064 TACO dataset problems
- `klear_code.parquet`: Klear code dataset
- `test_wo_prompt.parquet`: Test data without prompts

## Training Recipes

The repository includes several training recipes in `rl-recipe/recipe/`:

| Recipe | Description |
|--------|-------------|
| **DAPO** | Dynamic Advantage Policy Optimization |
| **PRIME** | Process Reinforcement through Implicit Rewards |
| **R1** | Reasoning training recipe |

## Code Judge

A code execution and evaluation service is included in `rl-recipe/code-judge/`. See its README for setup instructions.

## Citation

If you use this work, please cite:

```bibtex
@article{xcoder2024,
  title={X-Coder: Code Generation with Reinforcement Learning},
  author={IIGroup},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0.
