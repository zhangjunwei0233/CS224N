import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial

import pprint
pp = pprint.PrettyPrinter()

y = torch.ones(10, 5)
x = y + torch.randn_like(y)
pp.pprint(x)


class MultilayerPerceptron(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MultilayerPerceptron, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output


model = MultilayerPerceptron(5, 3, 5)

adam = optim.Adam(model.parameters(), lr=1e-1)

loss_function = nn.MSELoss()

y_pred = model(x)
pp.pprint(loss_function(y_pred, y).item())

n_epoch = 10

for epoch in range(n_epoch):
    # reset gradient
    adam.zero_grad()

    # Get model prediction
    y_pred = model(x)

    # calculate loss
    loss = loss_function(y_pred, y)

    # Print stats
    print(f"Epoch {epoch}: training loss: {loss}")

    # Compute the gradients
    loss.backward()

    # take a step
    adam.step()

# print(list(model.parameters()))
y_pred = model(x)
pp.pprint(y_pred)

# create a test data
x2 = y + torch.randn_like(y)
y_pred = model(x2)
pp.pprint(y_pred)


# ------- DEMO: Word Window Classification -------
corpus = [
    "We always come to Paris",
    "The professor is from Australia",
    "I live in Stanford",
    "He comes from Taiwan",
    "The capital of Turkey is Ankara"
]


# The preprocessing function we will use to generate our training examples
# Our function is a simple one, we lowercase the letters
# and then tokenize the words.
def preprocess_sentence(sentence):
    return sentence.lower().split()


train_sentences = [preprocess_sentence(sent) for sent in corpus]
# pp.pprint(train_sentences)

# Set of locations that appear in our corpus
locations = set(["australia", "ankara", "paris",
                "stanford", "taiwan", "turkey"])

# train labels
train_labels = [[1 if word in locations else 0 for word in sent]
                for sent in train_sentences]
# pp.pprint(train_labels)

vocabulary = set(w for s in train_sentences for w in s)
# pp.pprint(vocabulary)

# add an unknown and pad token
vocabulary.add("<unk>")
vocabulary.add("<pad>")


# Function that pads the given sentence
# We are introducing this function here as an example
# We will be utilizing it later in the tutorial
def pad_window(sentence, window_size, pad_token="<pad>"):
    window = [pad_token] * window_size
    return window + sentence + window

# window_size = 2
# pp.pprint(pad_window(train_sentences[0], window_size=window_size))


# We are just converting our vocabularly to a list to be able to index into it
# Sorting is not necessary, we sort to show an ordered word_to_ind dictionary
# That being said, we will see that having the index for the padding token
# be 0 is convenient as some PyTorch functions use it as a default value
# such as nn.utils.rnn.pad_sequence, which we will cover in a bit
idx_to_word = sorted(list(vocabulary))

word_to_idx = {word: ind for ind, word in enumerate(idx_to_word)}
# pp.pprint(word_to_idx)
# pp.pprint(ix_to_word[1])


def convert_token_to_indices(sentence, word_to_idx):
    return [word_to_idx.get(token, word_to_idx["<unk>"]) for token in sentence]

# example_sentence = ["we", "always", "come", "to", "kuwait"]
# example_indices = convert_token_to_indices(example_sentence, word_to_idx)
# restored_example = [idx_to_word[ind] for ind in example_indices]
# print(f"Original sentence is: {example_sentence}")
# print(f"Going from words to indices: {example_indices}")
# print(f"Going from indices to words: {restored_example}")

# example_padded_indices = [convert_token_to_indices(s, word_to_idx) for s in train_sentences]
# pp.pprint(example_padded_indices)


def custom_collate_fn(batch, window_size, word_to_ix):
    # Break out batch into the training examples (x) and labels (y)
    x, y = zip(*batch)

    # pad the examples
    x = [pad_window(s, window_size=window_size) for s in x]

    # Now we need to return words in out training examples to indices.
    x = [convert_token_to_indices(s, word_to_idx) for s in x]

    # We will now pad the examples so that the lengths of all the example in
    # one batch are the same, making it possible to do matrix operations.
    # We set the batch_first parameter to True so that the returned matrix has
    # the batch as the first dimension
    pad_token_ix = word_to_idx["<pad>"]

    # pad_sequence function expects the input to be a tensor
    x = [torch.LongTensor(x_i) for x_i in x]
    x_padded = nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=pad_token_ix)

    # pad the labels. Before doing so, we will need to record the number
    # of labels so that we know how many words existed in each example
    lengths = [len(label) for label in y]
    lengths = torch.tensor(lengths, dtype=torch.long)

    y = [torch.tensor(y_i, dtype=torch.long) for y_i in y]
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

    return x_padded, y_padded, lengths


data = list(zip(train_sentences, train_labels))
window_size = 2
collate_fn = partial(
    custom_collate_fn, window_size=window_size, word_to_ix=word_to_idx)

# Instantiate the DataLoader
loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Go trough one loop
# counter = 0
# for batched_x, batched_y, batched_lengths in loader:
#     print(f"Iteration {counter}")
#     print("Batched Input:")
#     print(batched_x)
#     print("Batched Labels:")
#     print(batched_y)
#     print("Batched Lengths:")
#     print(batched_lengths)
#     print("")
#     counter += 1

# print("Original Tensor:")
# print(batched_x)
# print("")
#
# # Create the 2 * 2 + 1 chunks
# chunk = batched_x.unfold(1, window_size * 2 + 1, 1)
# print("Windows: ")
# print(chunk)


class WordWindowClassifier(nn.Module):

    def __init__(self, hyperparameters, vocab_size, pad_ix=0):
        super(WordWindowClassifier, self).__init__()

        """ Instance variables """
        self.window_size = hyperparameters["window_size"]
        self.embed_dim = hyperparameters["embed_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.freeze_embeddings = hyperparameters["freeze_embeddings"]

        """ Embedding Layer
        Takes in a tensor contaning embedding indices, and returns the
        corresponding embeddings. The output is of dim
        (number_of_indices * embedding_dim).

        If freeze_embeddings is True, set the embedding layer parameters to be
        non-trainable. This is useful if we only want the parameter other than the
        embeddings parameters to changes.
        """
        self.embeds = nn.Embedding(
            vocab_size, self.embed_dim, padding_idx=pad_ix)
        if self.freeze_embeddings:
            self.embed_layer.weight.requires_grad = False

        """ Hidden Layer """
        full_window_size = 2 * window_size + 1
        self.hidden_layer = nn.Sequential(
            nn.Linear(full_window_size * self.embed_dim, self.hidden_dim),
            nn.Tanh()
        )

        """ Output Layer """
        self.output_layer = nn.Linear(self.hidden_dim, 1)

        """ Probabilities """
        self.probabilities = nn.Sigmoid()

    def forward(self, inputs):
        """
        Let B := batch_size
            L := window-padded sentence length
            D := self.embed_dim
            S := self.window_size
            H := self.hidden_dim

        inputs: a (B, L) tensor of token indices
        """
        B, L = inputs.size()

        """
        Reshaping.
        Takes in a (B, L) LongTensor
        Outputs a (B, L~, S) LongTensor
        """
        token_windows = inputs.unfold(1, 2 * self.window_size + 1, 1)
        _, adjusted_length, _ = token_windows.size()

        assert token_windows.size() == (B, adjusted_length, 2 * self.window_size + 1)

        """
        Embedding.
        Takes in a torch.LongTensor of size (B, L~, S)
        Outputs a (B, L~, S, D) FloatTensor
        """
        embedded_windows = self.embeds(token_windows)

        """
        Reshaping.
        Takes in a (B, L~, S, D) FloatTensor.
        Resizes it into a (B, L~, S * D) FloatTensor. (flatten all words in a single window to a vector)
        -1 argument "infers" what the last dimension should be based on leftover axes.
        """
        embedded_windows = embedded_windows.view(B, adjusted_length, -1)

        """
        Layer 1.
        Takes a (B, L~, S*D) FloatTensor.
        Resizes it into a (B, L~, H) FloatTensor.
        """
        layer_1 = self.hidden_layer(embedded_windows)

        """
        Layer 2.
        Takes in a (B, L~, H) FloatTensor.
        Resizes into a (B, L~, 1) FloatTensor.
        """
        output = self.output_layer(layer_1)

        """
        Softmax.
        Takes in a (B, L~, 1) FloatTensor of unnormalized class scores.
        Outputs a (B, L~) FloatTensor of (log-)normalized class scores.
        """
        output = self.probabilities(output)
        output = output.view(B, -1)

        return output


# Initialize a model
model_hyperparameters = {
    "batch_size": 4,
    "window_size": 2,
    "embed_dim": 25,
    "hidden_dim": 25,
    "freeze_embeddings": False,
}

vocab_size = len(word_to_idx)
model = WordWindowClassifier(model_hyperparameters, vocab_size)

# Define an optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Define a loss function, which computes to binary cross entropy loss
def loss_function(batch_outputs, batch_labels, batch_lenghts):
    bceloss = nn.BCELoss()
    loss = bceloss(batch_outputs, batch_labels.float())

    # Rescale the loss. Remember that we have useed lengths to store the
    # number of words in each training example
    loss = loss / batch_lenghts.sum().float()

    return loss


# Function that will be called in every epoch
def train_epoch(loss_function, optimizer, model, loader):

    # Keep track of total loss for the batch
    total_loss = 0
    for batch_inputs, batch_labels, batch_lengths in loader:
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass
        outputs = model.forward(batch_inputs)
        # Compute the batch loss
        loss = loss_function(outputs, batch_labels, batch_lengths)
        # Calculate the gradients
        loss.backward()
        # Update the parameters
        optimizer.step()
        total_loss += loss.item()

    return total_loss


# Function training our main traning loop
def train(loss_function, optimizer, model, loader, num_epochs=10000):

    # Iterate through each epoch and call our train_epoch function
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(loss_function, optimizer, model, loader)
        if epoch % 100 == 0:
            print(epoch_loss)


num_epochs = 1000
train(loss_function, optimizer, model, loader, num_epochs=num_epochs)


# Create test sentences
test_corpus = ["He come from China"]
test_sentences = [s.lower().split() for s in test_corpus]
test_labels = [[0, 0, 0, 1]]

# Create a test loader
test_data = list(zip(test_sentences, test_labels))
collate_fn = partial(custom_collate_fn, window_size=2, word_to_ix=word_to_idx)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

for test_instance, labels, _ in test_loader:
    outputs = model.forward(test_instance)
    print(labels)
    print(outputs)
