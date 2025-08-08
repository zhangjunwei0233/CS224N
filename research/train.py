import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from .config import DiffGPTConfig
from .modeling_diffgpt import DiffGPTForCausalLM
from .data import load_lm_dataset


@dataclass
class BoolFlag:
    @staticmethod
    def str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        return v.lower() in ("yes", "true", "t", "1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-v1")
    p.add_argument("--output_dir", default="/workspace/research/outputs/run")
    p.add_argument("--rope", type=BoolFlag.str2bool, default=True)
    p.add_argument("--research", type=BoolFlag.str2bool, default=False)
    p.add_argument("--impl_check", type=BoolFlag.str2bool, default=False)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--subset_ratio", type=float, default=0.02)
    p.add_argument("--eval_only", type=BoolFlag.str2bool, default=False)
    p.add_argument("--checkpoint", default=None)
    # Visualization and logging options
    p.add_argument("--log_tensorboard", type=BoolFlag.str2bool, default=True)
    p.add_argument("--log_figures", type=BoolFlag.str2bool, default=True)
    p.add_argument("--tb_flush_secs", type=int, default=10)
    # Optimization controls
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--optim", type=str, default="adamw_torch")
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", type=BoolFlag.str2bool, default=False)
    args = p.parse_args()

    # Fast implementation check mode: shrink model, data, and steps
    if args.impl_check:
        args.n_layer = min(args.n_layer, 2)
        args.n_head = min(args.n_head, 2)
        args.n_embd = min(args.n_embd, 64)
        args.block_size = min(args.block_size, 64)
        args.subset_ratio = min(args.subset_ratio, 0.001)
        args.batch_size = min(args.batch_size, 2)
        args.micro_batch_size = min(args.micro_batch_size, 1)
        args.num_train_epochs = 1

    tokenizer, lm_datasets = load_lm_dataset(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        block_size=args.block_size,
        subset_ratio=args.subset_ratio,
    )

    config = DiffGPTConfig(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        rope=args.rope,
        research=args.research,
    )

    model = DiffGPTForCausalLM(config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Configure reporting integrations
    report_to = (["tensorboard"] if args.log_tensorboard else ["none"])
    logging_steps = (1 if args.impl_check else 20)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=max(1, args.batch_size // args.micro_batch_size),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        eval_strategy=("no" if args.impl_check else "steps"),
        eval_steps=(5 if args.impl_check else 100),
        save_strategy=("no" if args.impl_check else "steps"),
        save_steps=(5 if args.impl_check else 200),
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_total_limit=2,
        max_steps=(10 if args.impl_check else -1),
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=report_to,
        save_safetensors=False,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Optional TensorBoard writer for custom scalars/histograms
    tb_writer = None
    if args.log_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=args.output_dir, flush_secs=args.tb_flush_secs)
        except Exception as e:
            print(f"[warn] Failed to initialize TensorBoard writer: {e}")
            tb_writer = None

    # Custom callback to log additional research-specific stats and figures
    from transformers import TrainerCallback, TrainerControl, TrainerState
    import time
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class ResearchLoggingCallback(TrainerCallback):
        def __init__(self, log_figures: bool = True):
            self.log_figures = log_figures
            self.train_start_time = None

        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            self.train_start_time = time.time()
            print({"event": "train_begin", "num_steps": state.max_steps, "num_epochs": args.num_train_epochs})
            if tb_writer is not None:
                tb_writer.add_text("run/info", f"Training started. steps={state.max_steps}, epochs={args.num_train_epochs}")

        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, model=None, **kwargs):
            step = state.global_step
            if logs is None:
                logs = {}
            # Console progress marker
            print({"event": "progress", "step": step, **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}})
            # TensorBoard scalars
            if tb_writer is not None:
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(f"metrics/{k}", float(v), step)
            # Research stats from model
            if getattr(model, "_last_diff_stats", None) is not None and tb_writer is not None:
                stats = model._last_diff_stats
                tb_writer.add_scalar("diff/mean_l2", stats["diff_mean_l2"], step)
                tb_writer.add_scalar("diff/std_l2", stats["diff_std_l2"], step)
                tb_writer.add_scalar("diff/max_l2", stats["diff_max_l2"], step)
                tb_writer.add_scalar("diff/min_l2", stats["diff_min_l2"], step)
                # Histogram sampling
                try:
                    tb_writer.add_histogram("diff/l2_hist", stats["diff_l2_hist_sample"], step)
                except Exception:
                    pass
            # Optional inline figure showing recent loss curve
            if self.log_figures and tb_writer is not None and "loss" in logs:
                fig = plt.figure(figsize=(4, 3))
                plt.plot([step], [logs["loss"],], marker="o")
                plt.title("Loss (last point)")
                plt.xlabel("step")
                plt.ylabel("loss")
                plt.tight_layout()
                tb_writer.add_figure("figures/loss_point", fig, step)
                plt.close(fig)

        def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, model=None, **kwargs):
            step = state.global_step
            if metrics is None:
                metrics = {}
            print({"event": "evaluate", "step": step, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}})
            if tb_writer is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(f"eval/{k}", float(v), step)
            # Save a small diagnostic figure comparing train vs eval loss if present
            if self.log_figures and tb_writer is not None and "eval_loss" in metrics:
                fig = plt.figure(figsize=(4, 3))
                plt.scatter([step], [metrics["eval_loss"]], marker="x", color="red")
                plt.title("Eval loss (point)")
                plt.xlabel("step")
                plt.ylabel("eval_loss")
                plt.tight_layout()
                tb_writer.add_figure("figures/eval_loss_point", fig, step)
                plt.close(fig)

        def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            elapsed = time.time() - self.train_start_time if self.train_start_time else None
            print({"event": "train_end", "elapsed_sec": float(elapsed) if elapsed is not None else None, "steps": state.global_step})
            if tb_writer is not None:
                tb_writer.add_text("run/info", f"Training ended. elapsed_sec={'{:.1f}'.format(elapsed) if elapsed is not None else 'NA'}")
                tb_writer.flush()

    training_args.logging_first_step = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
        callbacks=[ResearchLoggingCallback(log_figures=args.log_figures)],
    )

    if args.eval_only and args.checkpoint is not None:
        model = DiffGPTForCausalLM.from_pretrained(args.checkpoint)
        trainer.model = model
        eval_metrics = trainer.evaluate()
    else:
        trainer.train(resume_from_checkpoint=args.checkpoint)
        eval_metrics = trainer.evaluate()

    eval_loss = eval_metrics.get("eval_loss", None)
    if eval_loss is not None:
        ppl = float(math.exp(eval_loss)) if eval_loss < 20 else float("inf")
        print({"eval_loss": eval_loss, "perplexity": ppl})
        if tb_writer is not None:
            tb_writer.add_scalar("eval/perplexity", ppl, global_step=trainer.state.global_step)
    else:
        print(eval_metrics)

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()