from datasets import Dataset
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback, EarlyStoppingCallback
import torch
import numpy as np
import json

"""
--------------------------------------------------------------------
0. Preparing tokenizer and models
--------------------------------------------------------------------
"""
name = "distilbert/distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(name)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-cased", num_labels=2)

"""
--------------------------------------------------------------------
1. Loading a dataset
--------------------------------------------------------------------
"""
# DataLoader(zip(list1, list2))
dataset_name = "stanfordnlp/imdb"

imdb_dataset = load_dataset(dataset_name)


# Just take the first 50 tokens for speed/running on cpu
def truncate(example):
    return {
        'text': "".join(example['text'].split()[:50]),
        'label': example['label']
    }


# print(imdb_dataset)

# Take 128 random examples for train and 32 validation
small_imdb_dataset = DatasetDict(
    train=imdb_dataset['train'].shuffle(
        seed=1111).select(range(128)).map(truncate),
    val=imdb_dataset['train'].shuffle(
        seed=1111).select(range(128, 160)).map(truncate)
)
# print(small_imdb_dataset)
# print(small_imdb_dataset['train'][:10])

small_tokenized_dataset = small_imdb_dataset.map(
    lambda example: tokenizer(example['text'], padding=True, truncation=True),
    batched=True,
    batch_size=16
)

small_tokenized_dataset = small_tokenized_dataset.remove_columns(["text"])
small_tokenized_dataset = small_tokenized_dataset.rename_column(
    "label", "labels")
small_tokenized_dataset.set_format("torch")
# print(small_tokenized_dataset['train'][0:2])

train_dataloader = DataLoader(small_tokenized_dataset['train'], batch_size=16)
eval_dataloader = DataLoader(small_tokenized_dataset['val'], batch_size=16)


"""
--------------------------------------------------------------------
2. Training
--------------------------------------------------------------------
"""

"""
Hugging Face models are just torch.nn.Module, so you can train it
in a pytorch way.
"""
num_epochs = 1
num_training_steps = len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

best_val_loss = float("inf")
progess_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch_i, batch in enumerate(train_dataloader):

        # batch = ([text1, text2], [0, 1])
        output = model(**batch)

        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progess_bar.update(1)

    # validation
    model.eval()
    loss = 0
    for batch_i, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            output = model(**batch)
        loss += output.loss

    avg_val_loss = loss / len(eval_dataloader)
    print(f"Validation loss: {avg_val_loss}")
    if avg_val_loss < best_val_loss:
        print("Saving checkpoint!")
        best_val_loss = avg_val_loss
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'val_loss': best_val_loss,
        # },
        #     f"checkpoints/epoch_{epoch}.pt"
        # )

"""
Besides, Hugging Face offers a powerfull Trainer class to handel most needs.
The `Trainer` performs training, you can pass if the `TrainingArguments`, model,
the datasets, tokenizer, optimizer and even model checkpoints to resume training from.

After training, the `compute_metrics` function is called to calculate evaluation metrics.

It also allow you to write callbacks for early stopping, logging, ...

for more info on Training Arguments, See: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
for more info on callback, See: https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
"""
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-cased', num_labels=2)

arguments = TrainingArguments(
    output_dir='sample_hf_trainer',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    eval_strategy='epoch',  # run validation per epoch
    save_strategy='epoch',
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=224
)


def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['val'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


class LoggingCallBack(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


trainer.add_callback(EarlyStoppingCallback(
    early_stopping_patience=1, early_stopping_threshold=0.0))
trainer.add_callback(LoggingCallBack('sample_hf_trainer/log.jsonl'))

trainer.train()

results = trainer.predict(small_tokenized_dataset['val'])
print(results)

# To load our saved model, we can pass the path to the checkpoint into the `from_pretrained` method:
test_str = "I enjoyed the movie!"

finetuned_model = AutoModelForSequenceClassification.from_pretrained(
    "sample_hf_trainer/checkpoint-8")
model_inputs = tokenizer(test_str, return_tensors="pt")
prediction = torch.argmax(finetuned_model(**model_inputs).logits)
print(["NEGATIVE", "POSITIVE"][prediction])
