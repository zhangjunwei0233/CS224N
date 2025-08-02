import argparse
import random

import torch
from torch.utils.data import Dataset

# [part e]

# CharCorruptionDataset is a class that yields examples of a simplified
# span corruption objective.
# Implement the __getitem__ function based on the following instructions.
# Do not change the signature of the __init__ or __getitem__ functions.

# Make sure to implement the full spec for full credit -- we list below the
# criteria that must be satisfied for a full implementation.

# --------------
# Masking Specification

# The __getitem__ function takes an index and returns a data point (x, y) where
# x and y are Long tensors of length self.block_size. x encodes the input
# sequence, and y encodes the output sequence.

# 0. Use the idx argument of __getitem__ to retrieve the element of self.data
#    at the given index. We'll call the resulting data entry a document.

# 1. Randomly truncate the document to a length no less than 4 characters,
#    and no more than int(self.block_size*7/8) characters.

# - IMPORTANT: You are free to decide how to perform this random truncation, but
# make sure that the length is picked _randomly_ (every possible length from 4
# to int(self.block_size*7/8) has a chance of being picked) for full credit.

# 2. Now, break the (truncated) document into three substrings:
#
#     [prefix] [masked_content] [suffix]
#
#   In other words, choose three strings prefix, masked_content and suffix
#     such that prefix + masked_content + suffix = [the truncated document].
#   The length of [masked_content] should be random, and 1/4 the length of the
#     truncated document on average.

# - IMPORTANT: You are free to decide how to perform this operation, but
# make sure that the length is picked _randomly_ (has a chance of being more or
# less than 1/4 the length of the truncated document) for full credit.

# 3. Rearrange these substrings into the following form:
#
#     [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
#
#   This resulting string, denoted masked_string, serves as the output example.
#   Here MASK_CHAR is the masking character and [pads] is a string of repeated
#     PAD_CHAR characters chosen so that the entire string is of length
#     self.block_size + 1.
#   Intuitively, the string [masked_content] is removed from the document and
#     replaced with MASK_CHAR. After the suffix of the string, MASK_CHAR is seen
#     again, followed by the removed content, and the padding characters.

# 4. We now use masked_string to construct the input and output example pair. To
#    do so, simply take the input string to be masked_string[:-1], and the output
#    string to be masked_string[1:]. In other words, for each character, the goal
#    is to predict the next character in the masked string.

# 5. Making use of the vocabulary that you defined, encode the resulting input
#    and output strings as Long tensors and return the resulting data point.

# ----------------
# Here are some examples of input-output pairs (x, y):

#   x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇ and tran⁇ nalist, writer ⁇ □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: hatchig Mouradian. Khatchig Mouradian is a jour⁇ and tran⁇ nalist, writer ⁇ □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   x: Jaco⁇ enry ⁇ b H⁇ □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: aco⁇ enry ⁇ b H⁇ □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   x: John Stephen. Born in Glasgow, Steph⁇ lder's apprentice on⁇ en became a we⁇ □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: ohn Stephen. Born in Glasgow, Steph⁇ lder's apprentice on⁇ en became a we⁇ □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□


class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = "\u2047"  # the doublequestionmark character, for mask
        self.PAD_CHAR = "\u25A1"  # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print(f'data has {data_size} characters, {vocab_size} unique.')

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')  # split each line to a string
        self.data = self.data[:-1]  # remove the empty document

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        ### YOUR CODE HERE ###

        # Retrieve a document according to idx
        document = self.data[idx]

        # Randomly truncate the document to 4 <= len <= int(self.blocksize*7/8)
        min_length = 4
        max_length = int(self.block_size * 7 / 8)

        assert len(document) > 0, "received an empty document"
        if len(document) <= min_length:
            truncated_doc = document
        else:
            target_len = random.randint(
                min_length, min(len(document), max_length))
            # start_pos = random.randint(0, len(document) - target_len)
            start_pos = 0
            truncated_doc = document[start_pos:start_pos + target_len]

        # Break the truncated document into three substrings
        # [prefix] [masked_content] [suffix]
        # where masked_content length is 1/4 targetlen on average
        assert len(
            truncated_doc) >= 3, f"truncated document '{truncated_doc}' too short to separate into [prefix] [masked] [suffix]"

        masked_len_mean = len(truncated_doc) // 4
        min_masked_len = max(1, masked_len_mean // 2)
        max_masked_len = min(masked_len_mean * 3 // 2, len(truncated_doc) - 2)
        masked_len = random.randint(min_masked_len, max_masked_len)
        masked_start_pos = random.randint(
            1, len(truncated_doc) - masked_len - 1)

        prefix = truncated_doc[:masked_start_pos]
        masked_content = truncated_doc[masked_start_pos:masked_start_pos + masked_len]
        suffix = truncated_doc[masked_start_pos + masked_len:]

        # Rearrange the substring into following form:
        # [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + \
            masked_content + self.PAD_CHAR * \
            (self.block_size - len(truncated_doc) - 1)

        # Compute input and label string
        input_string = masked_string[:-1]
        output_string = masked_string[1:]

        # Encode the string as Long tensors and return
        x = torch.tensor([self.stoi[s]
                         for s in input_string], dtype=torch.long)
        y = torch.tensor([self.stoi[s]
                         for s in output_string], dtype=torch.long)

        return x, y
        ### END YOUR CODE ###


# The input-output pairs (x, y) of the NameDataset are of the following form:

# x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
# y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
# x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
# y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

# Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
# optimizing the model to predict the question, "Where was...".

# Note that the NameDataset should take the pretraining_dataset defined in run.py
# as an input. This is to allow the vocab specification of the NameDataset to be
# the same as that of the pretraining dataset.

# You don't need to implement anything in NameDataset.

class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        """Initialize

        Params:
            pretraining_dataset: CharCorruptionDataset instance
            data: (str) content of birth_places_train.tsv file
        """
        self.MASK_CHAR = "\u2047"  # the doublequestionmark character, for mask
        self.PAD_CHAR = "\u25A1"  # the empty square character, for pad
        self.itos = pretraining_dataset.itos  # dict[int, str]
        self.stoi = pretraining_dataset.stoi  # dict[str, int]
        self.block_size = pretraining_dataset.block_size  # int: 128
        # convert data into a list of str with "question\tanswer" format
        self.data = list(data.encode('utf-8').decode('ascii',
                         errors='ignore').split('\n'))

    def __len__(self):
        """Return the number of training samples (excludes the last empty line)
        """
        return len(self.data) - 1

    def __getitem__(self, idx):
        """
        Retrieve a sample according to its idx

        Returns:
            formatted input indices and output indices of this sample.
            input indices are of size (block_size-1,)
            output indices are of size (block_size,)
        """
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR  # fill in mask
        x = x + self.PAD_CHAR*(self.block_size - len(x))  # pad till block_size
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]

        x = x[:-1]  # remove the last token to form input tokens
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y


# Code below is strictly for your debugging purposes; feel free to modify
# as desired.

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
                      "Options: namedata, charcorruption.",
                      choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt', encoding='utf-8').read(), 128)
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
                                   open('birth_places_train.tsv', encoding='utf-8').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt', encoding='utf-8').read(), 128)
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError(
            f"Unknown dataset type in command line args: {args.dataset_type}")
