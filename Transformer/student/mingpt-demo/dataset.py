import math
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        """
        Initialize Dataset

        Paras:
            data: sentences that contains chars as unit element
            block_size: block size
        """
        chars = sorted(list(set(data)))  # Build vocabulary
        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size} characters, {vocab_size} unique.")
        print(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) charactors from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every charactor to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on the data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So far example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given just "he", please predict "l" next
        - given just "hel", please predict "l" next
        - given just "hell", please predict "o" next

        In addition, bacause the Dataloader will create batches of examples,
        every forward/backward pass during training will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y(B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the
        sequence along each batch dimension, and repeatedly always feed in a next
        charactor to get the next one.

        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a times, T times, with T forward
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
