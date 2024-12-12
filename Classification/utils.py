import torch
from torch import nn as nn
from typing import List, Tuple
import numpy as np


class Indexer(object):
    """
    Source for the Indexer class is the homework assignments
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


class PositionalEncoding(nn.Module):
    """
    Absolute positional encoding implementation. This defines embeddings for each position
    Source: Homework assignments
    """

    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        """
        Initialize the positional encoding module
        :param d_model: The embedding size
        :param num_positions: Max context length for the transformer
        :param batched: Is batching being used?
        """
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to the input embeddings
        :param x: The input embeddings
        :return: New embeddings with the positional embeddings added to the input embeddings
        """
        input_size = x.shape[-2]
        indices_to_embed = torch.arange(0, input_size).type(torch.LongTensor).to(x.device)
        if self.batched:
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


class SentimentExample(object):
    """
    This class defines a single sentiment example.
    It stores the expected input (with tokens separated by spaces as done in the SST dataset)
    It also stores the indices inside a tensor in input_tensor (and as an array in input_indexed),
    applying '<UNK>' as necessary
    The output is stored as both and integer and a scalar tensor. This is a label that is either 0-1
    or 0-5 depending on the task.
    """

    def __init__(self, input: str, output: int, vocab_index: Indexer, device='cpu'):
        self.input = input
        # Get the index of each token in the input:
        self.input_indexed = np.array(
            [vocab_index.index_of(token) if vocab_index.contains(token) else vocab_index.index_of("<UNK>") for token in
             input.split()])
        # Store indices in a tensor:
        self.input_tensor = torch.LongTensor(self.input_indexed).to(device)
        self.output = output
        # Store the output as a scalar tensor
        self.output_tensor = torch.tensor(self.output).to(device)

def pad_batch(batch: List[SentimentExample], pad_token: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Right pad a batch so that they are all the same length and create tensor of inputs and target values
    Compute an attention mask that can be used to prevent attending to pad tokens.
    :param batch: The batch of examples
    :param pad_token: The token index for '<PAD>' in the indexer
    :param device: The device to run on
    :return: The processed and padded batch of examples in tensor form, the targets for each example, and the attention mask
    """
    max_len = max(len(ex.input_tensor) for ex in batch)
    # Pad each input to the max length in the batch and create tensors for each example
    inputs = [torch.cat([ex.input_tensor, torch.full((max_len - len(ex.input_tensor),), pad_token).to(device)]) for ex in batch]
    # Create a tensor of target labels
    targets = torch.stack([ex.output_tensor for ex in batch]).to(device)
    # Compute a mask of ones and zeros to avoid attending to pad tokens (they are appropriately changed to -inf prior to computing softmax)
    mask = torch.stack(
        [torch.cat([torch.ones(len(ex.input_tensor)), torch.zeros(max_len - len(ex.input_tensor))]).to(device) for ex in batch])
    return torch.stack(inputs), targets, mask



