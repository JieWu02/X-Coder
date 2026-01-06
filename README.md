# X-Coder

- [SFT Training](#sft-training)
- [RLVR Training](#rlvr-training)
- [Dataset Description](#dataset-description)
- [Citation](#citation)

## SFT Training

### Environment Setup

```bash
pip install ms-swift -U
```

### Data Preparation

Download and convert the SFT training data from HuggingFace:

```bash
cd sft-recipe
python download_and_convert_data.py
```

This will download [IIGroup/X-Coder-SFT-376k](https://huggingface.co/datasets/IIGroup/X-Coder-SFT-376k) and convert it to `hybrid_376k.jsonl` format with `query` and `response` fields.

### Start SFT Training

For multi-node training (8 nodes x 8 GPUs):

```bash
# On each node, set the appropriate environment variables:
export NODE_RANK=<node_rank>  # 0, 1, 2, ..., 7
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500

cd sft-recipe
bash train_sft.sh
```

For single-node training, modify `train_sft.sh`:
- Set `NNODES=1`
- Adjust `CUDA_VISIBLE_DEVICES` as needed

---

## RLVR Training

### RLVR Quick Start

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

### RL Data Preparation

The rl training data (~17GB total) is hosted on HuggingFace: [IIGroup/X-Coder-RL-40k](https://huggingface.co/datasets/IIGroup/X-Coder-RL-40k)

#### Download RL Data

```bash
# Download all data (~17GB)
python download_data.py

# Or download only synthetic data (~8.5GB)
python download_data.py --syn-only
```

#### Data Structure

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

### Code Judge

A code execution and evaluation service is included in `rl-recipe/code-judge/`.

## Citation

If you find this work helpful, please cite:

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




