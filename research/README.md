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

## Visualization & Progress

- By default, TensorBoard logging is enabled. To disable: add `--log_tensorboard false`.
- Custom research diagnostics: when `--research true`, the model logs statistics of the differential token-embedding L2 norm (mean, std, min, max) and a histogram.
- Minimal inline figures for loss and eval loss points are written to TensorBoard under `figures/*`. Disable with `--log_figures false`.
- Console progress markers emit JSON-like dicts for `train_begin`, periodic `progress` (with current metrics), `evaluate`, and `train_end`.
- To view logs:
   
   ```
   tensorboard --logdir /workspace/research/outputs
   ```

### Example with more frequent logs

```
python -m research.train \
  --dataset wikitext --dataset_config wikitext-103-v1 \
  --output_dir /workspace/research/outputs/diff \
  --rope true --research true \
  --subset_ratio 0.01 --num_train_epochs 1
```

## Notes
- Set `--rope false` to use learned absolute positional embeddings in the baseline.
- The differential method is designed to work well with RoPE (`--rope true`).
- Use `--subset_ratio` to subsample the dataset for quick tests.
- All scripts use Hugging Face `Trainer` and `datasets`.