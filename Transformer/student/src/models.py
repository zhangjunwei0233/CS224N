"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier


Originally forked from Andrej Karpathy's minGPT.

CS224N 2023-24: Homework 4

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import attention

torch.manual_seed(0)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    rope = False
    research = False
    bottleneck_dim = None

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = attention.CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if not config.rope:
            self.pos_emb = nn.Parameter(torch.zeros(
                1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.rope = config.rope
        self.research = config.research
        # transformer
        self.blocks = nn.Sequential(*[Block(config)
                                    for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        # research
        self.start_emb = nn.Parameter(torch.zeros(1, 1, config.n_embd))

        print(
            f"number of parameters: {sum(p.numel() for p in self.parameters())}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward, model block size ({t}, {self.block_size}) is exhausted."

        # forward the GPT model
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)  # (b, t, d)

        # FIXME: test a new idea of sending in difference of tokens instead of tokens iteself
        if self.research:
            original_token_embeddings = token_embeddings

            token_embeddings_shifted = torch.roll(
                token_embeddings, shifts=1, dims=1)
            token_embeddings_shifted[:, 0, :] = self.start_emb

            token_embeddings = token_embeddings - token_embeddings_shifted

        if self.rope:
            x_input = token_embeddings
        else:
            # each position maps to a (learnable) vector
            position_embeddings = self.pos_emb[:, :t, :]
            x_input = token_embeddings + position_embeddings

        x = self.drop(x_input)
        x = self.blocks(x)
        x = self.ln_f(x)

        # FIXME: convert from semantic difference back to word
        # if transformer outputs x_t at position_t, that means this is the desired semantic change at this
        # position with attention to all the semantic changes before.
        # Thus, to get the desired next token output_{t+1}, we need to add x_t to input_t
        if self.research:
            x += original_token_embeddings

        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss
