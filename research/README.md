# Research: Differential Input Embedding Experiments

This folder contains a Hugging Face-compatible implementation of a GPT-style causal LM that replaces traditional positional embeddings with a differential method on input embedding sequences.

Two variants are supported:
- Baseline: Standard positional handling (learned absolute or RoPE).
- Differential: Uses token embedding deltas along the sequence and adds the base embedding back after the transformer stack.

## Quickstart

1) Install deps (prefer a venv):

```
pip install -r research/requirements.txt
```

2) Train on WikiText-103 (subset by default):

```
python -m research.train \
  --dataset wikitext --dataset_config wikitext-103-v1 \
  --output_dir /workspace/research/outputs/baseline \
  --rope true --research false
```

3) Train the differential variant:

```
python -m research.train \
  --dataset wikitext --dataset_config wikitext-103-v1 \
  --output_dir /workspace/research/outputs/diff \
  --rope true --research true
```

4) Evaluate perplexity (logged during training). You can also run:

```
python -m research.train --eval_only true --checkpoint /workspace/research/outputs/diff \
  --dataset wikitext --dataset_config wikitext-103-v1
```

## Notes
- Set `--rope false` to use learned absolute positional embeddings in the baseline.
- The differential method is designed to work well with RoPE (`--rope true`).
- Use `--subset_ratio` to subsample the dataset for quick tests.
- All scripts use Hugging Face `Trainer` and `datasets`.