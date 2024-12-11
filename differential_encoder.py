import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
from datasets import load_dataset
from typing import List
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import math
from traditional_encoder import Transformer
from typing import List, Tuple


def lambda_init_fn(depth: int) -> float:
    """
    The original differential transformer paper found a heuristic for setting lambda_init based on the
    layer depth, starting from 1. The formula is 0.8 - 0.6 * exp(-0.3 * depth) and is implemented here
    :param depth: The layer index, starting from 1
    :return: float
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DiffTransformer(Transformer):
    """
    A differential transformer is the same as a traditional one, but with Differential layers/attention instead.
    Therefore, we can take the same architecture through inheritance and replace the layers with DiffTransformerLayer instead.
    The projection for classification is done in the same way with pooling the final hidden states and then projecting to the number of classes.
    """
    def __init__(
            self,
            vocab_size: int,
            num_positions: int,
            d_model: int,
            d_internal: int,
            num_classes: int,
            num_layers: int,
            num_heads: int,
            hidden_size: int = 100
    ):
        """
        Initialize a differential transformer
        :param vocab_size: The size of the vocabulary as defined from the training set
        :param num_positions: Max context length of the transformer
        :param d_model: Embedding dimension
        :param d_internal: Internal dimension used for Q1, K1, Q2, K2, and V matrices (note that V will use twice this)
        :param num_classes: Number of classes to predict to
        :param num_layers: Number of transformer layers
        :param num_heads: Number of multi-head attention heads per layer
        :param hidden_size: Hidden size for the FFNN
        """
        super().__init__(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, num_heads, hidden_size)
        self.transformer_layers = nn.ModuleList(
            [DiffTransformerLayer(d_model, d_internal, num_heads, hidden_size, i + 1) for i in range(num_layers)]
        )


class DiffTransformerLayer(nn.Module):
    """
    This class defines a single layer in the differential transformer architecture.
    Note that we add the scalar lambda_init as well as the vectors lambda_q1, lambda_k1, lambda_q2, lambda_k2
    These are used for the reparamerization of the scalar lambda that is integral for differential attention
    """
    def __init__(self, d_model: int, d_internal: int, num_heads: int, hidden_size: int, depth: int):
        """
        Initialize a transformer layer
        :param d_model: Embedding dimension
        :param d_internal: Internal dimension used for Q1, K1, Q2, K2, and V matrices (note that V will use twice this)
        :param num_heads: Number of multi-head attention heads per layer
        :param hidden_size: Hidden size for the FFNN
        :param depth: We keep track of the depth of the layer, as this is used for computing lambda_init for each layer
        """
        super().__init__()

        self.num_heads = num_heads

        # Layer Norm to use after attention and FFNN
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # For the feed forward neural network
        self.FFLinear1 = nn.Linear(d_model, hidden_size)
        self.activation = nn.ReLU()
        self.FFLinear2 = nn.Linear(hidden_size, d_model)

        # These are the parameters used to compute lambda (this computation is done in the forward method)
        # lambda_init is heuristically defined by the depth of the specific layer
        # These parameters are shared across a whole layer
        self.lambda_init = lambda_init_fn(depth)
        # The vectors are all of length d_internal
        self.lambda_q1 = nn.Parameter(torch.zeros(d_internal, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(d_internal, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(d_internal, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(d_internal, dtype=torch.float32).normal_(mean=0, std=0.1))

        # We use differential multi-head attention instead of regular attention
        self.attention = DiffMultiHeadAttention(d_model, d_internal, num_heads, self.lambda_init)

    def forward(self, input_vecs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Do a forward pass through the Transformer Layer, including computing lambda to send to the attention heads
        :param input_vecs: The vectors to pass through this specific layer
        :param mask: Optional mask used for attention
        :return: The resulting tensor and a list of attention maps
        """
        # On a given pass, lambda is computed as exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init
        # This is specified in the paper
        lambda_current = (
            torch.exp(torch.dot(self.lambda_q1, self.lambda_k1)) -
            torch.exp(torch.dot(self.lambda_q2, self.lambda_k2)) +
            self.lambda_init
        )

        # Compute differential attention based on lambda_current:
        results, attention_maps = self.attention(input_vecs, lambda_current, mask)

        # Apply LayerNorm after attention and before the residual connection
        results = self.layer_norm1(input_vecs + results)

        # Go through FFNN:
        ff = self.FFLinear2(self.activation(self.FFLinear1(results)))

        # Apply LayerNorm after feed-forward and before the residual connection
        r2 = self.layer_norm2(results + ff)
        return r2, attention_maps


class DiffMultiHeadAttention(nn.Module):
    """
    This class defines a module for differential multi-head self-attention
    The major differences are applying layer norm to each head, and scaling the result by
    (1 - lambda_init) each time. This is recommended in the paper.
    """
    def __init__(self, d_model: int, d_internal: int, num_heads: int, lambda_init: float):
        """
        Initialize a differential multi-head attention block
        :param d_model: Embedding dimension
        :param d_internal: Internal dimension used for Q1, K1, Q2, K2, and V matrices (note that V will use twice this)
        :param num_heads: Number of multi-head attention heads per layer
        :param lambda_init: Value used to scale result of each head
        """
        super().__init__()

        # Define differential attention heads:
        self.attention_heads = nn.ModuleList(
            [DiffAttentionHead(d_model, d_internal) for _ in range(num_heads)])
        # Define the projection matrix. We multiply the first size by 2 because the V matrix is of size (N, 2 * d_internal)
        self.WO = nn.Linear(2 * num_heads * d_internal, d_model)

        # The layer norm to apply to each head as recommended in the paper:
        self.layer_norm = nn.LayerNorm(2 * d_internal)
        # The lambda init value used to scale the result from each head:
        self.lambda_init = lambda_init

    def forward(self, input_vecs: torch.Tensor, lambda_current: float, mask: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Do a forward pass through the differential attention block
        :param input_vecs: The vectors to pass through this specific layer
        :param lambda_current: The value lambda - this is used to scale one of the attention maps prior to taking the difference
        :param mask: Optional mask used for attention
        :return: The resulting tensor and a list of attention maps
        """
        results = []
        attention_maps = []

        # Pass through each differential attention head to get results
        for head in self.attention_heads:
            result, A = head(input_vecs, lambda_current, mask)  # lambda_current must be passed to each head
            result = self.layer_norm(result)  # Apply a layer norm to each result as recommended in the paper
            results.append(result * (1 - self.lambda_init))  # Multiply each result by (1 - lambda_init) before recording
            attention_maps.append(A)

        # Concatenate and project:
        results = torch.cat(results, dim=2)
        results = self.WO(results)

        return results, attention_maps


class DiffAttentionHead(nn.Module):
    """
    This class defines a single differential attention head.
    The major difference from standard self-attention is that two Q and two K matrices are computed.
    We can get two softmax attention maps, scale one of them by lambda_current, and then take the difference
    prior to multiplication by V.
    This is intended to reduce noise
    """
    def __init__(self, d_model: int, d_internal: int):
        """
        Initialize a differential attention head
        :param d_model: Embedding dimension
        :param d_internal: Internal dimension used for Q1, K1, Q2, K2, and V matrices (note that V will use twice this)
        """
        super().__init__()
        # Define two versions of WQ and WK to get two different attention maps
        # This can also be done by making a single WQ and a single WK that are both (d_model, 2 * d_internal)
        # and then cutting the result in half after the multiplication by the input to get two maps/
        # Note that V is size (d_model, 2 * d_internal) rather than just (d_model, d_internal)
        self.WQ1 = nn.Linear(d_model, d_internal)
        self.WK1 = nn.Linear(d_model, d_internal)
        self.WQ2 = nn.Linear(d_model, d_internal)
        self.WK2 = nn.Linear(d_model, d_internal)
        self.WV = nn.Linear(d_model, 2 * d_internal)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_vecs, lambda_current, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Do a forward pass through the attention_head. Note the additional lambda_current parameter
        :param input_vecs: The vectors to pass through this specific layer
        :param lambda_current: This is the layer-wide learnable scalar used to scale one of the attention maps prior to taking the difference
        :param mask: Optional mask used for attention
        :return: The resulting tensor and the attention map
        """
        # Compute two versions of Q and two versions of K:
        Q1 = self.WQ1(input_vecs)
        K1 = self.WK1(input_vecs)
        Q2 = self.WQ1(input_vecs)
        K2 = self.WK1(input_vecs)
        V = self.WV(input_vecs)

        # Compute two attention maps
        QK1 = torch.div(torch.matmul(Q1, K1.transpose(-2, -1)), self.WQ1.in_features ** 0.5)
        QK2 = torch.div(torch.matmul(Q2, K2.transpose(-2, -1)), self.WQ2.in_features ** 0.5)

        # Apply the mask to both maps if necessary:
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, QK1.size(1), -1)
            QK1 = QK1.masked_fill(mask_expanded == 0, float('-inf'))
            mask_expanded = mask.unsqueeze(1).expand(-1, QK2.size(1), -1)
            QK2 = QK2.masked_fill(mask_expanded == 0, float('-inf'))

        # Take the softmax of both maps
        A1 = self.softmax(QK1)
        A2 = self.softmax(QK2)

        # Scale one map by lambda and then take the difference
        # Store this result to return for potential attention map visualizations:
        A = A1 - (lambda_current * A2)

        # Finally, multiply by V
        Afinal = torch.matmul(A, V)

        return Afinal, A

