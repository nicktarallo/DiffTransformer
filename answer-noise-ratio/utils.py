import os
import random
import time
from matplotlib import pyplot as plt
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer
from math import sqrt

from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
import numpy as np

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32


class SimpleRMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization.
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super(SimpleRMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.norm(keepdim=True, dim=-1) / sqrt(self.d_model)
        result = (x / (rms + self.eps)) * self.scale
        #print(f"SimpleRMSNorm output shape: {result.shape}")
        return result
    

class FeedForward(nn.Module):
    """
    Implements the FeedForward network as used in transformers.
    """
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.linear1(x))
        #print(f"FeedForward linear1 output shape: {x.shape}")
        x = self.dropout(x)
        x = self.linear2(x)
        #print(f"FeedForward linear2 output shape: {x.shape}")
        return x
    


class OutputHead(nn.Module):
    """
    Implements the output layer for prediction.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super(OutputHead, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        result = self.linear(x)
        #print(f"OutputHead output shape: {result.shape}")
        return result
    

def validate(model, val_dataloader, criterion, device):
    # validation loop

    # Set model to evaluation mode
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = input_ids[:, 1:].to(device)
            
            logits = model(input_ids[:, :-1], mask=attention_mask[:, :-1])
            loss = criterion(logits.reshape(-1, model.vocab_size), 
                           target_ids.reshape(-1))
            total_val_loss += loss.item()
    
    return total_val_loss / len(val_dataloader)





def plot_training_progress(train_losses, val_losses, epochs_seen, folder_name):
    # plotting traning and validation losses
    plt.switch_backend('agg')
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 4))
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses vs epochs
    ax1.plot(epochs_seen, train_losses, label='Training Loss')
    ax1.plot(epochs_seen, val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    
    # Create unique filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'training_progress_{timestamp}.png'
    
    # Create plots directory if it doesn't exist

    filepath = os.path.join(folder_name, filename)
    
    plt.savefig(filepath, bbox_inches='tight')
    # plt.close()
    plt.close('all')  



def text_to_token_ids(text, tokenizer):
    # Encode text to token ids
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # Decode token ids to text
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Generate text with top-k sampling and temperature scaling"""

    model.eval()
    actual_context_size = min(context_size, model.max_seq_len)
    pad_token_id = model.embed.padding_idx
    device = idx.device
    # Remove leading padding tokens from the input sequence
    idx_trimmed = idx[:, (idx != pad_token_id).any(dim=0).nonzero().min().item():]

    for _ in range(max_new_tokens):
        idx_cond = idx_trimmed[:, -actual_context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            
            # Temperature scaling
            temperature = 0.7
            logits = logits / temperature
            
            # Top-k sampling with proper normalization
            top_k = 40
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Handle zero probability distributions
            if torch.any(torch.isnan(probs)):
                probs = torch.ones_like(probs) / probs.size(-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Break if pad token is generated
            if idx_next.item() == pad_token_id:
                break
                
            # Append the new token to the sequence
            idx_trimmed = torch.cat((idx_trimmed, idx_next), dim=1)
    
    return idx_trimmed



def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate text with proper sequence handling"""
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    
    # Encode and pad input
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    # Pad if needed
    if encoded.size(1) < model.max_seq_len:
        padding = torch.full((1, model.max_seq_len - encoded.size(1)),   tokenizer.pad_token_id, device=device)
        encoded = torch.cat([padding, encoded], dim=1)
    elif encoded.size(1) > model.max_seq_len:
        encoded = encoded[:, -model.max_seq_len:]
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=25,
            context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    
    model.train()





def create_needle_context(num_needles=4, context_length=500):
    """
    Creates context with target needle repeated at all depths plus distractor needles
    """
    # Create filler text
    filler_text = "This is a document about various cities and their numbers. "
    filler_words = filler_text.split()
    
    # Create base context
    context = [random.choice(filler_words) for _ in range(context_length)]
    
    # Create target needle
    target_city = "City_0"
    target_number = random.randint(100, 999)
    target_needle = f"The magic number for {target_city} is {target_number}"
    
    # Place target needle at all depths
    depths = [0.0, 0.25, 0.5, 0.75]
    target_positions = [int(depth * context_length) for depth in depths]
    for pos in target_positions:
        context[pos] = target_needle
    
    # Add distractor needles (N-1 additional needles)
    if num_needles > 1:
        available_positions = [i for i in range(context_length) 
                             if i not in target_positions]
        for i in range(1, num_needles):
            city = f"City_{i}"
            number = random.randint(100, 999)
            needle = f"The magic number for {city} is {number}"
            pos = random.choice(available_positions)
            context[pos] = needle
            available_positions.remove(pos)
    
    return " ".join(context), target_city, target_number




def plot_attention_analysis(persisted_results, epoch=0, folder_name='attn_maps'):
    depths = [0.0, 0.25, 0.5, 0.75]
    
    # Calculate averages across epochs
    avg_answer_scores = {
        depth: np.mean(persisted_results['answer_span'][depth])
        for depth in depths
    }
    avg_noise_scores = {
        depth: np.mean(persisted_results['noise_context'][depth])
        for depth in depths
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot averaged scores
    ax1.plot(depths, list(avg_answer_scores.values()), 'b-o', 
             linewidth=2, markersize=8)
    
    # Plot Answer Span Attention
    # ax1.plot(depths, answer_scores, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Context Depth')
    ax1.set_ylabel('Normalized Attention Score')
    ax1.set_title('Average Answer Span Attention')
    ax1.grid(True)
    
    ax2.plot(depths, list(avg_noise_scores.values()), 'r-o', 
             linewidth=2, markersize=8)
    
    # Plot Noise Context Attention
    # ax2.plot(depths, noise_scores, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Context Depth')
    ax2.set_ylabel('Normalized Attention Score')
    ax2.set_title('Average Noise Context Attention')
    ax2.grid(True)
    
    # Add main title
    plt.suptitle(f'Attention Analysis at Epoch {epoch}', fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'{folder_name}/attention_analysis_{timestamp}_epoch_{epoch}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def setup_tokenizer(name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def process_dataset(dataset, tokenizer, max_length=128):
    def tokenize_and_pad(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",  # Ensure padding up to max_length
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
       
    
    # Set format to PyTorch tensors
    tokenized = dataset.map(
        tokenize_and_pad, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return tokenized

def collate_fn(batch):
    """
    Custom collate function to handle dynamic padding.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    return {"input_ids": input_ids, "attention_mask": attention_mask}




def prepare_data(dataset_name, dataset_split, split):
    """
    Loads the dataset, tokenizes it, and prepares DataLoaders for training and validation.

    Args:
        dataset_name: Name of the dataset to load (e.g., "wikitext").
        dataset_split: Split of the dataset to load (e.g., "train[:5%]").
        tokenizer: Tokenizer for encoding text data.
        max_length: Maximum sequence length for tokenization.
        batch_size: Batch size for DataLoaders.
        train_split: Proportion of data to use for training (default: 0.9).
        collate_fn: Optional custom collate function for batching.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_split, split=split)
    tokenizer = setup_tokenizer()
    # Tokenize and process dataset
    tokenized_datasets = process_dataset(dataset, tokenizer, max_length=MAX_SEQ_LENGTH)
    train_split=0.9
    # Calculate split sizes
    total_size = len(tokenized_datasets)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    # Split into training and validation datasets
    train_dataset, val_dataset = random_split(tokenized_datasets, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader
