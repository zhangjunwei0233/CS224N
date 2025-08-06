'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW
from transformers import GPT2Tokenizer

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)

        # By default, fine-tune the full model.
        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        Cloze-style paraphrase detection: predict next token from full vocabulary.

        We structure the input as:
          'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

        The model predicts the next token, which should be "yes" or "no".
        Returns logits over the entire vocabulary for proper language modeling.
        """

        # Get GPT output with last token representation
        gpt_output = self.gpt(input_ids, attention_mask)
        # [batch_size, hidden_size]
        last_token_hidden = gpt_output['last_token']

        # Use weight-tied language modeling head to get logits over full vocabulary
        logits = self.gpt.hidden_state_to_token(
            last_token_hidden)  # [batch_size, vocab_size]

        return logits


def save_model(model, optimizer, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Step1: Load data
    # Create the data and its corresponding datasets and dataloader.
    para_train_data = load_paraphrase_data(
        args.para_train, split='train', max_size=args.max_train_size)
    para_dev_data = load_paraphrase_data(
        args.para_dev, split='dev', max_size=args.max_dev_size)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    # Step2: parse args and create model, optimizer
    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
    best_dev_acc = 0

    # Step3: Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Get the input and move it to the gpu (I do not recommend training this model on CPU).
            # Take first token of each label
            b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'][:, 0]
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            labels = labels.to(device)

            # Compute the loss, gradients, and update the model's parameters.
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            preds = torch.argmax(logits, dim=1)
            loss = F.cross_entropy(logits, labels, reduction='mean')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        dev_acc, dev_f1, * \
            _ = model_eval_paraphrase(para_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        print(
            f"Epoch {epoch}: train loss :: {train_loss:.3f}, dev acc :: {dev_acc:.3f}")


@torch.no_grad()
def test(args):
    """Evaluate your model on the dev and test datasets; save the predictions to disk."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath)

    model = ParaphraseGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model to test from {args.filepath}")

    para_dev_data = load_paraphrase_data(
        args.para_dev, split='dev', max_size=args.max_dev_size)
    para_test_data = load_paraphrase_data(
        args.para_test, split='test', max_size=args.max_test_size)

    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)
    para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_test_data.collate_fn)

    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(
        para_dev_dataloader, model, device)
    print(f"dev paraphrase acc :: {dev_para_acc:.3f}")
    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(
        para_test_dataloader, model, device)

    with open(args.para_dev_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p}, {s} \n")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--para_train", type=str,
                        default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str,
                        default="data/quora-test-student.csv")
    parser.add_argument("--para_dev_out", type=str,
                        default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str,
                        default="predictions/para-test-output.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--model_size", type=str,
                        help="The model size as specified on hugging face. DO NOT use the xl model.",
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

    # Dataset size limiting for debugging/validation
    parser.add_argument("--max_train_size", type=int, default=None,
                        help="Maximum number of training examples to use (for debugging)")
    parser.add_argument("--max_dev_size", type=int, default=None,
                        help="Maximum number of dev examples to use (for debugging)")
    parser.add_argument("--max_test_size", type=int, default=None,
                        help="Maximum number of test examples to use (for debugging)")

    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
    if args.model_size == 'gpt2':
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == 'gpt2-medium':
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == 'gpt2-large':
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise Exception(f'{args.model_size} is not supported.')
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train(args)
    test(args)
