import torch

from utils import PositionalEncoding
from typing import Tuple, List
import torch.nn as nn

class Transformer(nn.Module):
    """
    This class defines a regular transformer encoder model intended for classification with traditional self-attention.
    The classification is done by pooling the final hidden states and then projecting to the number of classes
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
        Initialize a transformer
        :param vocab_size: The size of the vocabulary as defined from the training set
        :param num_positions: Max context length of the transformer
        :param d_model: Embedding dimension
        :param d_internal: Internal dimension used for Q, K, and V matrices
        :param num_classes: Number of classes to predict to
        :param num_layers: Number of transformer layers
        :param num_heads: Number of multi-head attention heads per layer
        :param hidden_size: Hidden size for the FFNN
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_heads = num_heads
        self.d_model = d_model

        self.embeddings = nn.Embedding(vocab_size, d_model)

        # Use absolute positional encoding:
        self.positional_encoding = PositionalEncoding(d_model, num_positions, batched=True)

        # Transformer Layers
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(d_model, d_internal, num_heads, hidden_size) for _ in range(num_layers)]
        )

        # Pooling Options - used for classifying based on the last hidden states
        self.pooling_type = "mean"  # Options: "mean", "max", "cls"

        # Linear Layer for Classification
        self.linear = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)  # Apply softmax to output of the pooling operation

    def forward(self, indices: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Do a forward pass through the Transformer
        :param indices: The indices of each token in the example
        :param mask: Optional mask used for attention
        :return: The resulting tensor and a list of attention maps
        """
        # Embedding and positional encoding
        result = self.embeddings(indices)
        result = self.positional_encoding(result)

        # Pass through Transformer Layers
        attention_maps = []
        for layer in self.transformer_layers:
            result, A = layer(result, mask)
            attention_maps.extend(A)

        # Pooling over the hidden last hidden states prior to classifiction
        if self.pooling_type == "mean":
            result = result.mean(dim=1)  # Mean pooling across sequence length
        elif self.pooling_type == "max":
            result, _ = result.max(dim=1)  # Max pooling across sequence length
        elif self.pooling_type == "cls":
            result = result[:, 0, :]  # CLS token (assuming first token is CLS)

        # Classification
        result = self.linear(result)  # Result shape: (batch_size, num_classes)
        result = self.log_softmax(result)
        return result, attention_maps

    def predict(self, indices: torch.Tensor, mask: torch.Tensor = None) -> torch.LongTensor:
        """
        Make a prediction
        :param indices: The indices of each token in the example
        :param mask: Optional mask used for attention
        :return: The predicted label
        """
        result, _ = self.forward(indices, mask)
        prediction = torch.argmax(result, dim=1)  # Predictions from the pooled output
        return prediction


class TransformerLayer(nn.Module):
    """
    This class defines a single layer in the transformer architecture.
    """
    def __init__(self, d_model: int, d_internal: int, num_heads: int, hidden_size: int):
        """
        Initialize a transformer layer
        :param d_model: Embedding dimension
        :param d_internal: Internal dimension used for Q, K, and V matrices
        :param num_heads: Number of multi-head attention heads per layer
        :param hidden_size: Hidden size for the FFNN
        """
        super().__init__()

        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)


        self.attention = MultiHeadAttention(d_model, d_internal, num_heads)

        # Layer norms to use after attention and FFNN
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # For the FFNN:
        self.FFLinear1 = nn.Linear(d_model, hidden_size)
        self.activation = nn.ReLU()
        self.FFLinear2 = nn.Linear(hidden_size, d_model)

    def forward(self, input_vecs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Do a forward pass through the Transformer Layer
        :param input_vecs: The vectors to pass through this specific layer
        :param mask: Optional mask used for attention
        :return: The resulting tensor and a list of attention maps
        """
        results, attention_maps = self.attention(input_vecs, mask)

        # Apply LayerNorm after attention and before the residual connection
        results = self.layer_norm1(input_vecs + results)

        ff = self.FFLinear2(self.activation(self.FFLinear1(results)))

        # Apply LayerNorm after feed-forward and before the residual connection
        r2 = self.layer_norm2(results + ff)
        return r2, attention_maps


class MultiHeadAttention(nn.Module):
    """
    This class defines a module for standard multi-head self-attention
    """
    def __init__(self, d_model: int, d_internal: int, num_heads: int):
        """
        Initialize a multi-head attention block
        :param d_model: Embedding dimension
        :param d_internal: Internal dimension used for Q, K, and V matrices
        :param num_heads: Number of multi-head attention heads per layer
        """
        super().__init__()

        self.attention_heads = nn.ModuleList([AttentionHead(d_model, d_internal) for _ in range(num_heads)])
        self.WO = nn.Linear(num_heads * d_internal, d_model)  # The projection matrix

    def forward(self, input_vecs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Do a forward pass through the multi-head attention block
        :param input_vecs: The vectors to pass through this specific layer
        :param mask: Optional mask used for attention
        :return: The resulting tensor and a list of attention maps
        """
        attention_maps = []
        results = []

        # Pass through each head and record the resulting vectors
        for head in self.attention_heads:
            result, A = head(input_vecs, mask)
            results.append(result)
            attention_maps.append(A)

        # Concatenate the vectors and then project back to d_model
        results = torch.cat(results, dim=2)
        results = self.WO(results)

        return results, attention_maps


class AttentionHead(nn.Module):
    """
    This class defines a single attention head for use in standard self-attention
    """
    def __init__(self, d_model, d_internal):
        """
        Initialize a self-attention head
        :param d_model: Embedding dimension
        :param d_internal: Internal dimension used for Q, K, and V matrices
        """
        super().__init__()

        self.WQ = nn.Linear(d_model, d_internal)
        self.WK = nn.Linear(d_model, d_internal)
        self.WV = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_vecs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Do a forward pass through the attention_head
        :param input_vecs: The vectors to pass through this specific layer
        :param mask: Optional mask used for attention
        :return: The resulting tensor and the attention map
        """
        # Compute Q, K, and V
        Q = self.WQ(input_vecs)
        K = self.WK(input_vecs)
        V = self.WV(input_vecs)

        # Compute Q * K transpose and divide by the scalar
        QK = torch.div(torch.matmul(Q, K.transpose(-2, -1)), self.WQ.in_features ** 0.5)

        # Apply attention mask if necessary:
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, QK.size(1), -1)
            QK = QK.masked_fill(mask_expanded == 0, float('-inf'))

        # Apply softmax and multiplication by V
        # Retain A after doing softmax to return attention map for potential plottnig
        A = self.softmax(QK)
        Afinal = torch.matmul(A, V)

        return Afinal, A