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
  --output_dir /root/autodl-tmp/research/outputs/baseline \
  --rope true --research false
```

3) Train the differential variant:

```
python -m research.train \
  --dataset wikitext --dataset_config wikitext-103-v1 \
  --output_dir /root/autodl-tmp/research/outputs/diff \
  --rope true --research true
```

4) Evaluate perplexity (logged during training). You can also run:

```
python -m research.train --eval_only true --checkpoint /root/autodl-tmp/research/outputs/diff \
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
  --output_dir /root/autodl-tmp/research/outputs/diff \
  --rope true --research true \
  --subset_ratio 0.01 --num_train_epochs 1
```

## Training Tutor

### 小样本过拟合检验——确定模型与损失没有实现性错误，并找到稳定学习率区间

目的：找到并稳定学习率区间。

```bash
python -m research.train \
    --dataset wikitext --dataset_config wikitext-103-v1 \
    --output_dir /root/autodl-tmp/research/outputs/sanity \
    --subset_ratio 0.001 --num_train_epochs 10 \
    --n_later 2 --n_head 2 --n_embd 128 --block_size 128 \
    --batch_size 16 --micro_batch_size 2 \
    --learning_rate 1e-3 --lr_scheduler_type consine --warmup_steps 50 \
    --weight_decay 0.1 --adam_beta2 0.95 --max_grad_norm 1.0 \
    --rope true --research false
```

观察，如果loss不降低或者发散，先把学习率降低10倍再试；仍不行则把dropout降到更低，或者缩小batch。

### 中等规模稳定训练——验证超参数

目的：找到能稳步下降的训练曲线和合理的PPL区间。

```bash
python -m research.train \
    --dataset wikitext --dataset_config wikitext-103-v1 \
    --output_dir /root/autodl-tmp/research/outputs/baseline_mid \
    --subset_ratio 0.2 --num_train_epochs 1 \
    --n_later 12 --n_head 12 --n_embd 768 --block_size 1024 \
    --batch_size 64 --micro_batch_size 4 \
    --learning_rate 1e-4 --lr_scheduler_type consine --warmup_ratio 0.03 \
    --weight_decay 0.1 --adam_beta2 0.95 --max_grad_norm 1.0 \
    --gradient_checkpointing true \
    --rope true --research false
```

- 学习率：3e-4 对小有效batch常偏大，优先 1e-4/5e-5；若前500–1k步loss不降，直接再降10倍。
- Warmup：warmup_ratio 0.03–0.1 或 warmup_steps 1k–3k 都可试。
- 有效batch：用梯度累积把 batch_size/micro_batch_size 提高到 32–128 的有效规模。
- 梯度裁剪：max_grad_norm=1.0 保留；若仍偶发爆炸，可降到 0.5。
- 调度器：cosine 比 constant 更稳；也可试 linear。
- 观察 TensorBoard：loss、学习率、梯度（已内置标记），若曲线锯齿大，通常是 LR 偏高或 acum 太小。

### 全量训练

```bash
python -m research.train \
    --dataset wikitext --dataset_config wikitext-103-v1 \
    --output_dir /root/autodl-tmp/research/outputs/baseline \
    --subset_ratio 1.0 --num_train_epochs 1 \
    --n_later 12 --n_head 12 --n_embd 768 --block_size 1024 \
    --batch_size 128 --micro_batch_size 4 \
    --learning_rate 1e-4 --lr_scheduler_type consine --warmup_ratio 0.03 \
    --weight_decay 0.1 --adam_beta2 0.95 --max_grad_norm 1.0 \
    --gradient_checkpointing true \
    --rope true --research false
```

### 差分策略训练建议

先用 baseline（research=false）跑通并拿到合理的 PPL，再开启 --research true 对比。

开启差分后，通常把 LR 再降一档（例如 5e-5），其余保持一致。


## Notes
- Set `--rope false` to use learned absolute positional embeddings in the baseline.
- The differential method is designed to work well with RoPE (`--rope true`).
- Use `--subset_ratio` to subsample the dataset for quick tests.
- All scripts use Hugging Face `Trainer` and `datasets`.
