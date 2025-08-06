# !/usr/bin/env python3

"""
Evaluation code for Quora paraphrase detection.

model_eval_paraphrase is suitable for the dev (and train) dataloaders where the label information is available.
model_test_paraphrase is suitable for the test dataloader where label information is not available.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from transformers import GPT2Tokenizer
from datasets import (
    SonnetsDataset,
)

TQDM_DISABLE = False


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
    # Switch to eval model, will turn off randomness like dropout.
    model.eval()

    # Get tokenizer to convert between token IDs and binary predictions
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    yes_token_id = tokenizer.encode("yes")[0]
    no_token_id = tokenizer.encode("no")[0]

    y_true, y_pred, sent_ids = [], [], []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        # Take first token
        b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch[
            'attention_mask'], batch['sent_ids'], batch['labels'][:, 0]

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask).cpu().numpy()
        pred_token_ids = np.argmax(logits, axis=1)

        # Convert predicted token IDs to binary predictions (0=no, 1=yes)
        binary_preds = [1 if token_id ==
                        yes_token_id else 0 for token_id in pred_token_ids]

        # Convert true label token IDs to binary (0=no, 1=yes)
        binary_labels = [1 if token_id ==
                         yes_token_id else 0 for token_id in labels.cpu().numpy()]

        y_true.extend(binary_labels)
        y_pred.extend(binary_preds)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
    # Switch to eval model, will turn off randomness like dropout.
    model.eval()

    # Get tokenizer to convert token IDs to binary predictions
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    yes_token_id = tokenizer.encode("yes")[0]
    no_token_id = tokenizer.encode("no")[0]

    y_pred, sent_ids = [], []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask).cpu().numpy()
        pred_token_ids = np.argmax(logits, axis=1)

        # Convert predicted token IDs to binary predictions (0=no, 1=yes)
        binary_preds = [1 if token_id ==
                        yes_token_id else 0 for token_id in pred_token_ids]

        y_pred.extend(binary_preds)
        sent_ids.extend(b_sent_ids)

    return y_pred, sent_ids


@torch.no_grad()
def eval_sonnet_dev(model, held_out_dataset, gold_path='data/TRUE_sonnets_held_out_dev.txt', temperature=1.2, top_p=0.9):
    """
    Evaluate sonnet generation on dev set during training.

    Args:
        model: The SonnetGPT model
        held_out_dataset: The held-out sonnet dataset (first 3 lines)
        gold_path: Path to true complete sonnets
        temperature: Generation temperature
        top_p: Top-p sampling parameter

    Returns:
        CHRF score as float
    """
    device = model.get_device()
    chrf = CHRF()

    # Generate sonnets for all held-out examples
    generated_sonnets = []
    model.eval()

    for batch in held_out_dataset:
        # Get the partial sonnet (first 3 lines)
        encoding = model.tokenizer(
            batch[1], return_tensors='pt', padding=False, truncation=True).to(device)

        # Generate completion
        output = model.generate(
            encoding['input_ids'], temperature=temperature, top_p=top_p)

        # Decode the complete generated sonnet
        full_sonnet = output[1]  # output[1] is the decoded string
        generated_sonnets.append(full_sonnet)

    # Load true sonnets
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # Compute CHRF score
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)
