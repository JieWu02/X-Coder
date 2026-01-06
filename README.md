# RLVR

## Table of Contents

- [Quick Start](#quick-start)
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

# 4. Download rl training data
cd ..
python download_data.py

# 5. Start training
cd rl-recipe
bash train_scripts/install.sh
bash train_scripts/xcoder-rl-train.sh
```

## Data Preparation

The rl training data (~17GB total) is hosted on HuggingFace: [IIGroup/X-Coder-RL-40k](https://huggingface.co/datasets/IIGroup/X-Coder-RL-40k)

### Download Data

```bash
# Download all data (~17GB)
python download_data.py

# Or download only synthetic data (~8.5GB)
python download_data.py --syn-only
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

## Code Judge

A code execution and evaluation service is included in `rl-recipe/code-judge/`. See its README for setup instructions.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{
anonymous2025xcoder,
title={X-Coder: Advancing Competitive Programming with Fully Synthetic Tasks, Solutions, and Tests},
author={Anonymous},
booktitle={Submitted to The Fourteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=jp4dzBilqH},
note={under review}
}
```

## License

This project is licensed under the Apache License 2.0.


