import torch
import torch.nn.functional as F
from torch import nn, Tensor
from math import sqrt
from datasets import load_dataset
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from utils import MAX_SEQ_LENGTH, collate_fn, prepare_data, \
    process_dataset, setup_tokenizer, validate, \
        plot_attention_analysis, plot_training_progress \
            ,generate_and_print_sample, analyze_attention_scores \
                ,OutputHead, FeedForward, SimpleRMSNorm
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('xla')


class DiffAttn(nn.Module):
    """
    Differential Attention module with learnable lambda.
    """
    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, 2 * d)
        self.W_k = nn.Linear(embedding_dim, 2 * d)
        self.W_v = nn.Linear(embedding_dim, d)  # Project to d dimensions to match attention output
        self.lambda_ = nn.Parameter(torch.randn(1))  # Scalar learnable lambda
        self.lambda_init = 0.05

    def forward(self, X: Tensor, mask:Tensor=None) -> Tensor:
        batch_size, seq_len, _ = X.shape
        # if torch.isnan(X).any():
        #     print(f"NaN detected in diff attn block. X Shape: {X.shape}")
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        
    
        Q1, Q2 = self.split(Q)
        K1, K2 = self.split(K)
        
      
        # Add numerical stability
        s = 1 / sqrt(max(self.d, 1))  # Prevent division by zero
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * s
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * s

    
        
        if mask is not None:
            mask = mask.to(torch.bool)
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
            A1 = A1.masked_fill(mask, float('-inf'))
            A2 = A2.masked_fill(mask, float('-inf'))
        
        A1 = A1 - A1.max(dim=-1, keepdim=True).values
        A2 = A2 - A2.max(dim=-1, keepdim=True).values
        # Add numerical stability to softmax
        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)

        # Ensure stability by replacing NaNs
        A1_softmax = torch.nan_to_num(A1_softmax, nan=0.0, posinf=1.0, neginf=0.0)
        A2_softmax = torch.nan_to_num(A2_softmax, nan=0.0, posinf=1.0, neginf=0.0)
        if torch.isnan(A1_softmax).any():
            print(f"NaN detected in diff attn block. A1_softmax Shape: {A1_softmax.shape}")
        # Clamp lambda to prevent explosion
        lambda_ = torch.exp(self.lambda_) + self.lambda_init
        
        differential_attn = A1_softmax - lambda_ * A2_softmax
        result = torch.matmul(differential_attn, V)

        return result
    
    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention module.
    """
    def __init__(self, h: int, d_head: int, embedding_dim: int, lambda_init: float):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.h = h
        self.d_head = d_head
        self.lambda_init = lambda_init
        self.embedding_dim = embedding_dim
        self.diff_attn_heads = nn.ModuleList([DiffAttn(d_head, embedding_dim) for _ in range(h)])
        self.W_o = nn.Linear(h * d_head, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, X: Tensor, mask:Tensor=None) -> Tensor:
        O_list = [head(X, mask) for head in self.diff_attn_heads]
        O_concat = torch.cat(O_list, dim=-1)
        if torch.isnan(O_concat).any():
            print(f"NaN detected inmultihead atten  block. O_list Shape: {O_concat.shape}")

        #print(f"MultiHead O_concat shape: {O_concat.shape}")
        result = self.W_o(O_concat)
        if torch.isnan(result).any():
            print(f"NaN detected inmultihead atten  block. result Shape: {result.shape}")
        #print(f"MultiHead W_o output shape: {result.shape}")
        result = self.norm(result)
        if torch.isnan(result).any():
            print(f"NaN detected inmultihead atten  block. Shape: {result.shape}")
        #print(f"MultiHead norm output shape: {result.shape}")
        result = result * (1 - self.lambda_init)
        if torch.isnan(result).any():
            print(f"NaN detected inmultihead atten  block. Shape: {result.shape}")
        return result


class DifferentialTransformerBlock(nn.Module):
    """
    Implements a Differential Transformer Block.
    """
    def __init__(self, d_model: int, heads: int = 12, dropout: float = 0.1, lambda_init: float = 0.05):
        super(DifferentialTransformerBlock, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout
        self.lambda_init = lambda_init
        head_dim = d_model // heads

        self.attn = MultiHeadDifferentialAttention(heads, head_dim, d_model, lambda_init)
        self.ffn = FeedForward(d_model, d_model * 4, dropout)
        self.norm = SimpleRMSNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        residual = x
        attended = self.attn(self.norm(x), mask) + residual
        #print(f"DifferentialTransformerBlock first attention output shape: {attended.shape}")
        residual_two = attended
        attended = self.ffn(self.norm(residual_two)) + residual_two
        #print(f"DifferentialTransformerBlock second attention output shape: {attended.shape}")
        if torch.isnan(attended).any():
                print(f"NaN detected in diffrential tranformer block. Shape: {attended.shape}")
        return attended


class DifferentialTransformer(nn.Module):
    """
    Implements a full Differential Transformer.
    """
    def __init__(self, d_model: int = 3072, n_heads: int = 12, d_head:int = 2, dropout: float = 0.1, lambda_init: float = 0.8, n_layers: int = 24, vocab_size: int = 30000, max_seq_len: int = 128, padding_idx: int = None, tokenizer=None):
        super(DifferentialTransformer, self).__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.heads = n_heads
        self.dropout = dropout
        self.lambda_init = lambda_init
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_head = d_head
        self.dropout_layer = nn.Dropout(self.dropout)


        # Position Embedding
        self.position_embedding = nn.Embedding(max_seq_len, self.d_model)

        # Layer initialization
        self.layers = nn.ModuleList([DifferentialTransformerBlock(self.d_model, self.heads, self.dropout, lambda_init) for _ in range(self.n_layers)])
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx=padding_idx)
        self.norm = SimpleRMSNorm(self.d_model)
        self.output_head = OutputHead(self.d_model, vocab_size)
      
    def forward(self, input_ids: Tensor, mask: Tensor = None) -> Tensor:
        seq_len = min(input_ids.size(1), self.max_seq_len)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        x = self.embed(input_ids) + self.position_embedding(positions)
        x = self.dropout_layer(x)
        
        # Create default attention mask if none provided
        if mask is None:
            # Create padding mask based on pad_token_id
            mask = (input_ids != self.embed.padding_idx).to(input_ids.device)
        else:
            mask = mask.bool()
        # Create causal mask
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), 
            diagonal=1
        ).bool()
        
        # Combine with padding mask
        attention_mask = mask.unsqueeze(1).unsqueeze(2)
        attention_mask = torch.logical_and(attention_mask, ~causal_mask.unsqueeze(0))
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=attention_mask)

        x = self.norm(x)
        output = self.output_head(x)
        return output


def get_attention_maps(model, tokenizer, context, query):
    """
    Get attention maps for a given context and query.
    """


    # Set model to evaluation mode
    model.eval()

    # Tokenize input
    inputs = tokenizer(context + query, return_tensors="pt").to(device)
    
    # Get embeddings
    embeddings = model.embed(inputs.input_ids)
    
    # Get attention scores
    attention_maps = []
    with torch.no_grad():
        for layer in model.layers:
            # Use the first head (or iterate for multiple heads)
            head = layer.attn.diff_attn_heads[0]
            
            # Compute queries and keys
            Q1, Q2 = head.split(head.W_q(embeddings))
            K1, K2 = head.split(head.W_k(embeddings))
            
            # Compute attention scores
            s = 1 / sqrt(head.d)
            A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * s
            A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * s
            
            # Apply mask if necessary
            if inputs.attention_mask is not None:
                mask = inputs.attention_mask.unsqueeze(1).unsqueeze(2).bool()
                A1 = A1.masked_fill(~mask, float('-inf'))
                A2 = A2.masked_fill(~mask, float('-inf'))
            
            # Apply softmax
            A1_softmax = F.softmax(A1, dim=-1)
            A2_softmax = F.softmax(A2, dim=-1)
            
            # # Compute differential attention
            # lambda_ = torch.exp(head.lambda_) + head.lambda_init
            # diff_attn = A1_softmax - lambda_ * A2_softmax

            # Before computing differential attention
            lambda_ = torch.exp(head.lambda_) + head.lambda_init
            diff_attn = A1_softmax - lambda_ * A2_softmax

            # Normalize the differential attention to [-1, 1] range
            diff_attn = 2 * (diff_attn - diff_attn.min()) / (diff_attn.max() - diff_attn.min()) - 1
            
            attention_maps.append(diff_attn)
    def normalize_attention_maps(attention_maps):
        normalized_maps = []
        for attn_map in attention_maps:
            # Global normalization across all dimensions
            attn_flat = attn_map.view(-1)
            min_val = torch.min(attn_flat)
            max_val = torch.max(attn_flat)
            
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            normalized_map = (attn_map - min_val) / (max_val - min_val + eps)
            normalized_maps.append(normalized_map)
        return normalized_maps

    norm_atn = normalize_attention_maps(attention_maps)
    return norm_atn

  

def train_model(model, train_dataloader, val_dataloader, num_epochs=20, learning_rate=1e-4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean').to(device)
    
    epochs_seen = []
    tokens_seen = []
    train_losses = []
    val_losses = []
    persisted_attn_results = {
        'answer_span': {depth: [] for depth in [0.0, 0.25, 0.5, 0.75]},
        'noise_context': {depth: [] for depth in [0.0, 0.25, 0.5, 0.75]}
    }

    # Initialize tracking variables
    best_val_loss = float('inf')
    needle_results = []
    
    # Create results directory
    os.makedirs('diff_checkpoints', exist_ok=True)
    os.makedirs('diff_attn_maps', exist_ok=True)
    os.makedirs('diff_plots', exist_ok=True)


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = input_ids[:, 1:].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids[:, :-1], mask=attention_mask[:, :-1])
            if torch.isnan(logits).any():
                print(f"NaN detected in logits. Shape: {logits.shape}")
                continue
            # Reshape logits and targets properly
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.reshape(-1, vocab_size)
            
            target_ids = target_ids.reshape(-1)
            loss = criterion(logits, target_ids)
            if torch.isnan(loss):
                print(f"NaN loss detected. Logits min/max: {logits.min():.4f}/{logits.max():.4f}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

           
            epoch_loss += loss.item()
            epoch_tokens += input_ids.numel()
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        val_loss = validate(model, val_dataloader, criterion, device)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        epochs_seen.append(epoch + 1)
        tokens_seen.append(epoch_tokens)
        
        if (epoch + 1) % 5 == 0:

            attn_results = analyze_attention_scores(model, tokenizer, epoch, num_samples=50, context_length=seq_len, folder_name='diff_attn_maps')
            for depth in [0.0, 0.25, 0.5, 0.75]:
                persisted_attn_results['answer_span'][depth].append(
                    attn_results['avg_answer_attention_per_depth'][depth]
                )
                persisted_attn_results['noise_context'][depth].append(
                    attn_results['avg_noise_attention_per_depth'][depth]
                )
            plot_attention_analysis(persisted_attn_results, epoch, 'diff_attn_maps')
            # Plot progress and generate sample
            plot_training_progress(train_losses, val_losses, epochs_seen, tokens_seen, 'diff_plots')
        # Save training progress plot
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': avg_train_loss,
                'needle_results': needle_results
            }, 'diff_checkpoints/diff_best_model.pt')
            
        
       
        sample_text = "The quick brown fox"
        generate_and_print_sample(model, tokenizer, device, sample_text)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    print(f"Using device: {device}")
    print("Loading dataset and tokenizer...")

    tokenizer = setup_tokenizer("gpt2")
    # Prepare DataLoaders
    train_dataloader, val_dataloader = prepare_data(
        dataset_name="wikitext",
        dataset_split="wikitext-2-raw-v1",
        split="train[:5%]"
    )

    # Hyperparameter optimization
    try:
         # Initialize model with fixed hyperparameters
        model = DifferentialTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=512,  # Fixed size
            n_heads=8,    # Fixed number of heads
            d_head=64,    # d_model // n_heads
            n_layers=4,   # Fixed number of layers
            max_seq_len=MAX_SEQ_LENGTH,
            dropout=0.1,
            padding_idx=tokenizer.pad_token_id,
            tokenizer=tokenizer
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of parameters: {total_params}")
        # Train model
        trained_model, train_losses, val_losses = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=20,
            learning_rate=1e-4
        )

        torch.save(trained_model.state_dict(), 'diff_final_model.pt')


    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise