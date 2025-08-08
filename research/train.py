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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=max(1, args.batch_size // args.micro_batch_size),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        eval_strategy=("no" if args.impl_check else "steps"),
        eval_steps=(10 if args.impl_check else 200),
        save_strategy=("no" if args.impl_check else "steps"),
        save_steps=(10 if args.impl_check else 200),
        logging_strategy="steps",
        logging_steps=(1 if args.impl_check else 50),
        save_total_limit=1,
        max_steps=(10 if args.impl_check else -1),
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=["none"],
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
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
    else:
        print(eval_metrics)


if __name__ == "__main__":
    main()