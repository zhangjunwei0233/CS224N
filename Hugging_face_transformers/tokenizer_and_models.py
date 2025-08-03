from transformers import DistilBertConfig, DistilBertModel
from transformers import AutoModel, AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertModel
from transformers import DistilBertTokenizer, DistilBertTokenizerFast, AutoTokenizer
import warnings

from matplotlib import pyplot as plt
import torch

warnings.filterwarnings("ignore", category=FutureWarning)


def print_encoding(model_inputs, indent=4):
    indent_str = " " * indent
    print("{")
    for k, v in model_inputs.items():
        print(indent_str + k + ":")
        print(indent_str + indent_str + str(v))
    print("}")


"""
--------------------------------------------------------------------
1. Below are the COMMON WORKFLOW for using Hugging face transformers
--------------------------------------------------------------------
"""
tokenizer = AutoTokenizer.from_pretrained(
    "siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained(
    "siebert/sentiment-roberta-large-english")

inputs = "I'm excited to learn about Hugging Face Transformers!"
tokenized_inputs = tokenizer(inputs, return_tensors="pt")
outputs = model(**tokenized_inputs)

labels = ['NEGATIVE', 'POSITIVE']
prediction = torch.argmax(outputs.logits)

print("Input:")
print(inputs)
print()
print("Tokenized Inputs:")
print_encoding(tokenized_inputs)
print()
print("Model Outputs:")
print(outputs)
print()
print(f"The prediction is {labels[prediction]}")


"""
---------------------------------------------------------------------------------------
2. Tokenizers
---------------------------------------------------------------------------------------

Tokenizers take raw strings and output the model input dictionary (indices, masks, ...)

Pretrained models are implemented along with their tokenizers

You can access tokenizers either with the Tokenizer class specific to the model you use,
or with the AutoTokenizer class
"""
name = "distilbert/distilbert-base-cased"  # use DistilBert tokenizer for example
# name = "user/name" when loading from
# name = local_path when using save_pretrained() method

tokenizer = DistilBertTokenizer.from_pretrained(name)  # Written in Python
print(tokenizer)
tokenizer = DistilBertTokenizerFast.from_pretrained(name)  # Written in Rust
print(tokenizer)
tokenizer = AutoTokenizer.from_pretrained(name)  # convenient! Defaults to Fast
print(tokenizer)


"""
This is how you call the tokenizer
"""
input_str = "Hugging Face Transformers is great!"
tokenized_inputs = tokenizer(input_str)

print("Vanilla Tokenizaton")
print_encoding(tokenized_inputs)
print()


"""
Tokenization happens in a few steps:
    1. tokenize the input via `tokenizer.tokenize()`
    2. convert tokens to ids via `tokenizer.convert_tokens_to_ids()`
    3. add special tokens via string cancatenation

    the indice sequence can be decode via `tokenizer.decode()`
"""
cls = [tokenizer.cls_token_id]
sep = [tokenizer.sep_token_id]
# print(f"cls: {cls}, sep: {sep}")

input_tokens = tokenizer.tokenize(input_str)
input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
input_ids_special_tokens = cls + input_ids + sep
decoded_str = tokenizer.decode(input_ids_special_tokens)

print("start:                ", input_str)
print("tokenize:             ", input_tokens)
print("convert_tokens_to_ids:", input_ids)
print("add special tokens:   ", input_ids_special_tokens)
print("--------")
print("decode:               ", decoded_str)
print("")


"""
For Fast Tokenizers (Rust implemented version), there's an option
to do all these steps with one cmd: `tokenizer._tokenizer.encode()`
"""
inputs = tokenizer._tokenizer.encode(input_str)

print(input_str)
print("--------")
print(f"Number of tokens:    {len(inputs)}")
print(f"Ids:                 {inputs.ids}")
print(f"Tokens:              {inputs.tokens}")
print(f"Special tokens mask: {inputs.special_tokens_mask}")
print("--------")
print("char_to_word locates a token that contains the given charactor idx")
char_idx = 8
print(f"For example, the {char_idx + 1}th character of the string is '{input_str[char_idx]}'," +
      f" and it's part of wordpiece {inputs.char_to_token(char_idx)}, '{inputs.tokens[inputs.char_to_token(char_idx)]}'")

"""
Tokenizer can return pytorch tensors!
"""
model_inputs = tokenizer(
    "Hugging Face Transformers is great!", return_tensors="pt")
print("PyTorch Tensors:")
print_encoding(model_inputs)

"""
You can pass multiple string into to tokenizer and pad them as you need
"""
model_inputs = tokenizer(["Hugging Face Transformers is great!",
                          "The quick brown fox jumps over the lazy dog." +
                          "Then the dog got up and ran away because she didn't like foxes."],
                         return_tensors="pt",
                         padding=True,
                         truncation=True)
print(
    f"Pad token: {tokenizer.pad_token} | Pad token id: {tokenizer.pad_token_id}")
print("Padding:")
print_encoding(model_inputs)

"""
You can also decode a whole batch at once
"""
print("Batch Decode:")
print(tokenizer.batch_decode(model_inputs.input_ids))
print()
print("Batch Decode: (no special characters)")
print(tokenizer.batch_decode(model_inputs.input_ids, skip_special_tokens=True))


"""
--------------------------------------------------------------------
3. Models
--------------------------------------------------------------------

the AutoModel method has lots of pretrained models of all architecture (base model).
You just need to specify the task you want to do, ant it will pick the right model
(task specific model that contains newly initialized weights) for you to do post training.

A full list of choices are available here https://huggingface.co/docs/transformers/model_doc/auto
"""
print("Loading base model")
base_model = DistilBertModel.from_pretrained('distilbert-base-cased')
print("Loading classification model from base model's checkpoint")
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-cased', num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-cased', num_labels=2)

"""
You can also choose to initialize the model totally with random weights
"""
# Initialize a DistilBERT configuration
configuration = DistilBertConfig()
configuration.num_labels = 2
# Initialize a model (with random weights) from the configuration
model = DistilBertForSequenceClassification(configuration)

# Access the model configuration
configuration = model.config
print(configuration)

"""
Models take inputs as keyword arguments

and the keyword tokenizer returns are the same as the
keyword arguments the model expects, so we can directly use
`**` to unpack the returned dict to input args
"""
model_inputs = tokenizer(input_str, return_tensors='pt')
model_outputs = model(**model_inputs)

print()
print_encoding(model_inputs)
print()
print(model_outputs)
print()
print(
    f"Distribution over labels: {torch.softmax(model_outputs.logits, dim=1)}")

"""
These models are just pytorch modules, which means you can calculate loss,
do backward_prop or check parameters just as in pytorch!
"""
# Calculate the loss as normal
label = torch.tensor([1])  # Targetted output
loss = torch.nn.functional.cross_entropy(model_outputs.logits, label)
print(f"Loss: {loss}")
loss.backward()

# Get the parameters
print(list(model.named_parameters())[0])

"""
Hugging Face also provides an additional easy way to calculate loss,
that is, to pass in a label
"""
model_inputs = tokenizer(input_str, return_tensors="pt")

labels = ['NEGATIVE', 'POSITIVE']
model_inputs['labels'] = torch.tensor([1])  # add labels item into input dict

model_outputs = model(**model_inputs)

print(model_outputs)
print()
print(f"Model predictions: {labels[model_outputs.logits.argmax()]}")


"""
With hugging face, you can visualize hidden states and attention weight very easily!
"""
model = AutoModel.from_pretrained(
    "distilbert-base-cased", output_attentions=True, output_hidden_states=True)
model.eval()

model_inputs = tokenizer(input_str, return_tensors="pt")
with torch.no_grad():
    model_outputs = model(**model_inputs)

print("Hidden state size (per layer):  ", model_outputs.hidden_states[0].shape)
print("Attention head size (per layer):", model_outputs.attentions[0].shape)

tokens = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
print(tokens)

n_layers = len(model_outputs.attentions)
n_heads = len(model_outputs.attentions[0][0])
fig, axes = plt.subplots(6, 12)
fig.set_size_inches(18.5*2, 10.5*2)
for layer in range(n_layers):
    for i in range(n_heads):
        axes[layer, i].imshow(
            model_outputs.attentions[layer][0, i].detach().numpy())
        axes[layer][i].set_xticks(list(range(9)))
        axes[layer][i].set_xticklabels(labels=tokens, rotation="vertical")
        axes[layer][i].set_yticks(list(range(9)))
        axes[layer][i].set_yticklabels(labels=tokens)

        if layer == 5:
            axes[layer, i].set(xlabel=f"head={i}")
        if i == 0:
            axes[layer, i].set(ylabel=f"layer={layer}")

plt.subplots_adjust(wspace=0.3)
plt.show()
