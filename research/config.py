from transformers import PretrainedConfig


class DiffGPTConfig(PretrainedConfig):
    model_type = "diffgpt"

    def __init__(
        self,
        vocab_size: int = 50257,
        block_size: int = 1024,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        rope: bool = True,
        research: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.rope = rope
        self.research = research