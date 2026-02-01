# Feature Tree Data

This folder contains the feature artifacts used by X-Coder question generation.

## Files

- `feature_config_sampled.jsonl`
  - Full feature config dataset used by question generation (tracked by Git LFS in this repo).
  - If you did not download LFS objects, this may be a small pointer file instead of real data.
  - Format: each line is a `{"features": ..., "idx": ..., "leaf_count": ...}` record compatible with
    `question_generation/scripts/generate_questions.py`.

- `feature_evoled.json`
  - A merged, tree-structured feature taxonomy produced by the `extract + evol + merge` pipeline.
  - Used as the default input tree for `question_generation/features_trees_data/utils/sample_feature_configs.py`.

- `feature_extracted.jsonl`
  - Per-problem extracted records (JSONL) produced by the `extract` stage.

## Prepare `feature_config_sampled.jsonl` (from EpiCoder outputs)

In this repo, `feature_config_sampled.jsonl` is tracked by Git LFS. If you did not download LFS objects,
the file will be a small pointer and question generation will fail to load features.

You have two options:

1) Install git-lfs and pull the real file:
```bash
git lfs install
git lfs pull
```

2) Re-generate a compatible JSONL locally from a merged feature tree (recommended for open-source reproducibility):
```bash
cd data-recipe
python question_generation/features_trees_data/utils/sample_feature_configs.py \
  --output /tmp/feature_config_sampled_100.jsonl \
  --num 100 \
  --seed 42 \
  --min-leaf-count 25 \
  --shapes 3,3,1 3,3,2 4,3,2
```
By default, the script prefers the vendored merged feature tree
`question_generation/features_trees_data/feature_evoled.json` (if present), and otherwise auto-detects from a
sibling `../EpiCoder` checkout.
If your EpiCoder path is different, pass `--epicoder-root /path/to/EpiCoder` or `--input-tree /path/to/*_merged_fea.json` (or `*_merged_fea_ori.json`).

Then point question generation to the generated file:
```bash
python question_generation/scripts/generate_questions.py \
  --features-file /tmp/feature_config_sampled_100.jsonl \
  --min-leaf-count 25 \
  --start 0 --end 100 \
  --output ../output/questions_0_100.jsonl \
  --batch-size 16
```

## (Optional) Rebuild the merged feature tree from scratch (extract + evol)

If you want to reproduce the **feature tree construction** (the extract/evol prompts are part of our method),
you can run the following pipeline inside `data-recipe/question_generation/`:

1) Configure Azure OpenAI (TRAPI-style) env vars (see `/home/v-jiewu5/call_api.py` for the credential pattern):
```bash
export AZURE_OPENAI_ENDPOINT="https://trapi.research.microsoft.com/gcr/shared"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-5_2025-08-07"
export AZURE_OPENAI_TOKEN_SCOPE="api://trapi/.default"
```

2) Run feature extraction on your seed dataset (JSONL/JSON with `question` + optional `solutions`):
```bash
python question_generation/features_trees_data/utils/extract_features_epicoder.py \
  --input /path/to/seed.jsonl \
  --output /tmp/extract_features.jsonl \
  --start 0 --end 100
```

3) Merge extracted features into a base tree:
```bash
python question_generation/features_trees_data/utils/merge_extracted_features_tree.py \
  --input /tmp/extract_features.jsonl \
  --output /tmp/base_feature_tree.json
```

4) Evolve (expand) the tree with LLM:
```bash
python question_generation/features_trees_data/utils/evolve_feature_tree_epicoder.py \
  --input-tree /tmp/base_feature_tree.json \
  --output /tmp/feature_evol.jsonl \
  --num-steps 100 \
  --seed 42
```

5) Merge evolved expansions back into a merged tree:
```bash
python question_generation/features_trees_data/utils/merge_evolved_feature_tree.py \
  --base-tree /tmp/base_feature_tree.json \
  --evol-jsonl /tmp/feature_evol.jsonl \
  --output /tmp/merged_feature_tree.json
```

6) Sample 100 configs from the merged tree (compatible with question generation):
```bash
python question_generation/features_trees_data/utils/sample_feature_configs.py \
  --input-tree /tmp/merged_feature_tree.json \
  --output /tmp/feature_config_sampled_100.jsonl \
  --num 100 --seed 42 --min-leaf-count 25 \
  --shapes 3,3,1 3,3,2 4,3,2
```
