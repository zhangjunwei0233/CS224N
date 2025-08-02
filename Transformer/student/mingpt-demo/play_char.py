from mingpt.utils import sample
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.model import GPT, GPTConfig
from mingpt.utils import set_seed
import logging
import torch
from dataset import CharDataset

# set up Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# make deterministic
set_seed(42)

# Create training dataset
block_size = 128  # spatial extent of the model for its context
text = open('input.txt', 'r').read()
train_dataset = CharDataset(text, block_size)

# Create the model with proper config
mconf = GPTConfig(train_dataset.vocab_size,
                  train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

# Create trainer and train the model
tconf = TrainerConfig(max_epochs=2, batch_size=64, learning_rate=6e-4, lr_decay=True,
                      warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size, num_workers=0)

trainer = Trainer(model, train_dataset, None, tconf)
# print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")
# trainer.train()  # This runs too slow on cpu

# sample some character-level Shakespeare
context = "O God, O God!"
x = torch.tensor([train_dataset.stoi[s] for s in context],
                 dtype=torch.long)[None, ...].to(trainer.device)
y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)
