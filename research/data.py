from typing import Dict, Optional

from datasets import load_dataset
from transformers import AutoTokenizer


def get_tokenizer(name: str = "gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lm_dataset(
    dataset: str = "wikitext",
    dataset_config: Optional[str] = "wikitext-103-v1",
    tokenizer_name: str = "gpt2",
    block_size: int = 1024,
    subset_ratio: float = 0.05,
):
    ds = load_dataset(dataset, dataset_config)
    tokenizer = get_tokenizer(tokenizer_name)

    def tokenize_function(examples: Dict[str, list]):
        return tokenizer(examples["text"])  # returns input_ids, attention_mask

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds["train"].column_names)

    def group_texts(examples):
        # Concatenate texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True)

    if 0 < subset_ratio < 1.0:
        def select_split(split_name):
            split = lm_datasets[split_name]
            num = max(1, int(len(split) * subset_ratio))
            return split.select(range(num))
        lm_datasets = {
            "train": select_split("train"),
            "validation": select_split("validation") if "validation" in lm_datasets else select_split("test"),
        }
    else:
        lm_datasets = {"train": lm_datasets["train"], "validation": lm_datasets.get("validation", lm_datasets.get("test"))}

    return tokenizer, lm_datasets