
# Differential Transformer for Machine Translation

This repository implements a differential transformer architecture for machine translation tasks. The model introduces custom attention mechanisms and follows the standard encoder-decoder framework. It is inspired by the Transformer architecture with additional innovations to enhance performance.

## Features

- **Differential Attention Mechanism:** Implements a custom attention mechanism with layer-specific learnable parameters.
- **Positional Encoding:** Adds sequence positional information to embeddings.
- **Encoder-Decoder Framework:** Utilizes stacked layers of encoders and decoders.
- **Custom Feed-Forward Layers:** Includes fully connected layers with dropout and activation functions.
- **Projection Layer:** Outputs a probability distribution over the target vocabulary.

## Installation

Ensure you have Python 3.8 or later installed. Install the required dependencies:

```bash
pip install torch sacrebleu datasets
```

## Running the Notebook

1. Clone the repository and navigate to the project directory:
   ```bash
   git https://github.com/nicktarallo/DiffTransformer.git
   cd DiffTransformer/Machine-Translation
   ```

2. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook Differential_Transformer_MT.ipynb
   ```

3. Follow the step-by-step instructions in the notebook to:
   - Load the dataset
   - Tokenize and preprocess the data
   - Initialize the model
   - Train the model
   - Evaluate the model

## Key Components

### 1. Input Embeddings

Converts input token IDs into vector embeddings.

### 2. Positional Encoding

Adds positional information to embeddings using sine and cosine functions.

### 3. Differential Attention Block

A custom attention mechanism with layer-specific lambda parameters.

### 4. Encoder and Decoder

- **Encoder:** Stacks self-attention and feed-forward blocks.
- **Decoder:** Adds cross-attention in addition to self-attention.

### 5. Differential Transformer

Combines the encoder and decoder to construct the complete model.

## Model Configurations

To run and test the model, use the following configurations in the `build_differential_transformer()` function:

| Configuration | d_model | H (Heads) | N (Layers) | FFWD (Feed-Forward Dimension) |
|---------------|---------|-----------|------------|-------------------------------|
| Config 1      | 256     | 8         | 6          | 1024                          |
| Config 2      | 128     | 4         | 3          | 512                           |
| Config 3      | 64      | 2         | 1          | 256                           |

Example:

```python
model = build_differential_transformer(
    src_vocab_size=50000,
    tgt_vocab_size=50000,
    src_seq_len=100,
    tgt_seq_len=100,
    d_model=256,
    N=6,
    h=8,
    d_ff=1024
)
```


## Evaluation

Evaluate the model using BLEU scores from the `sacrebleu` package:

```python
from sacrebleu import corpus_bleu

bleu = corpus_bleu(predictions, [references])
print(f"BLEU Score: {bleu.score}")
```

## Contributing

Contributions are welcome! Please submit pull requests with clear documentation of changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

