import json
import time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from math import sqrt
from tqdm import tqdm
import os
from torch.amp import GradScaler, autocast



from utils import create_needle_context, prepare_data, \
        setup_tokenizer, validate, \
        plot_attention_analysis, plot_training_progress \
            ,generate_and_print_sample\
                ,OutputHead, FeedForward, MAX_SEQ_LENGTH
   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('xla')

# Initialize GradScaler
scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    """
    Attention module
    """
    def __init__(self, d: int, embedding_dim: int):
        super(Attention, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, d)
        self.W_k = nn.Linear(embedding_dim, d)
        self.W_v = nn.Linear(embedding_dim, d)

    def forward(self, X: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, seq_len, _ = X.shape
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        s = 1 / sqrt(max(self.d, 1))
        A = torch.matmul(Q, K.transpose(-2, -1)) * s

        if mask is not None:
            mask = mask.to(torch.bool)
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
            A = A.masked_fill(mask, float('-inf'))

        A = A - A.max(dim=-1, keepdim=True).values
        A = F.softmax(A, dim=-1)
        A = torch.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)

        return torch.matmul(A, V)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    """
    def __init__(self, h: int, d_head: int, embedding_dim: int):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_head = d_head
        self.embedding_dim = embedding_dim
        self.attention_heads = nn.ModuleList([Attention(d_head, embedding_dim) for _ in range(h)])
        self.W_o = nn.Linear(h * d_head, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, X: Tensor, mask: Tensor = None) -> Tensor:
        O_list = [head(X, mask) for head in self.attention_heads]
        O_concat = torch.cat(O_list, dim=-1)
        result = self.W_o(O_concat)
        return self.norm(result)

class TransformerBlock(nn.Module):
    '''
    Transformer layer block.
    
    '''
    def __init__(self, d_model: int, heads: int = 12, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.heads = heads
        head_dim = d_model // heads

        self.attn = MultiHeadAttention(heads, head_dim, d_model)
        self.ffn = FeedForward(d_model, d_model * 4, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        results = self.attn(self.norm(x), mask) + x
        attended = self.ffn(self.norm(results)) + results
        return attended
    
class DecoderTransformer(nn.Module):
    """
    Implements a full Transformer.
    """
    def __init__(self, d_model: int = 3072, n_heads: int = 12, d_head:int = 2, dropout: float = 0.1, lambda_init: float = 0.8, n_layers: int = 24, vocab_size: int = 30000, max_seq_len: int = 128, padding_idx: int = None, tokenizer=None):
        super(DecoderTransformer, self).__init__()
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
        self.layers = nn.ModuleList([TransformerBlock(self.d_model, self.heads, self.dropout) for _ in range(self.n_layers)])
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx=padding_idx)
        self.norm = nn.LayerNorm(self.d_model)
        self.output_head = OutputHead(self.d_model, vocab_size)
      
    def forward(self, input_ids: Tensor, mask: Tensor = None) -> Tensor:
        seq_len = min(input_ids.size(1), self.max_seq_len)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        x = self.embed(input_ids) + self.position_embedding(positions)* sqrt(self.d_model)
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
        
    model.eval()

    # Tokenize input
    inputs = tokenizer(context + query, return_tensors="pt").to(device)
    
    # Get embeddings
    embeddings = model.embed(inputs.input_ids)
    
    # Get attention scores
    attention_maps = []
    with torch.no_grad():
        for layer in model.layers:
            # Use the attention head
            head = layer.attn.attention_heads[0]
            
            # Compute queries, keys, and values
            Q = head.W_q(embeddings)
            K = head.W_k(embeddings)
            
            
            # Compute attention scores
            s = 1 / sqrt(head.d)
            A = torch.matmul(Q, K.transpose(-2, -1)) * s
            
            # Apply mask if necessary
            if inputs.attention_mask is not None:
                mask = inputs.attention_mask.unsqueeze(1).unsqueeze(2).bool()
                A = A.masked_fill(~mask, float('-inf'))
            
            # Apply softmax
            A = A - A.max(dim=-1, keepdim=True).values
            A_softmax = F.softmax(A, dim=-1)
            A_softmax = torch.nan_to_num(A_softmax, nan=0.0, posinf=1.0, neginf=0.0)
            
            attention_maps.append(A_softmax)

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



def analyze_attention_scores(model, tokenizer, epoch, num_samples=50, context_length=124, folder_name='diff_attn_maps'):
    '''
    Analyze attention scores for target needle span and noise context.
    '''
    model.eval()
    attention_analysis = {
        'answer_span_scores_per_depth': {depth: [] for depth in [0.0, 0.25, 0.5, 0.75]},
        'noise_context_scores_per_depth': {depth: [] for depth in [0.0, 0.25, 0.5, 0.75]},
    }

    depths = [0.0, 0.25, 0.5, 0.75]
    context_len_for_analysis = context_length//4
    for _ in range(num_samples):
        # Create context with target at specified depth (only once per sample)
        context, target_city, target_number = create_needle_context(
            num_needles=4,  # Fixed number of needles
            context_length=context_len_for_analysis
        )
        
        query = f"What is the magic number for {target_city}?"
        
        # Get attention maps (only once per sample)
        attention_maps = get_attention_maps(model, tokenizer, context, query)
        attention_map = attention_maps[-1][0].cpu()  # Use last layer
        
        # Find target needle span (same across all depths for this sample)
        target_needle = f"The magic number for {target_city} is {target_number}"
        target_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(target_needle)['input_ids']
        )
        
        # Calculate attention scores for each depth
        for depth in depths:
            # Calculate target position based on depth
            target_pos = int(depth * context_len_for_analysis)
            target_end = target_pos + len(target_tokens)
            
            # Calculate normalized attention scores
            # 1. Answer span attention
            answer_attention = attention_map[:, target_pos:target_end].mean().item()
            
            # 2. Noise context attention (everything except answer span)
            noise_mask = torch.ones_like(attention_map)
            noise_mask[:, target_pos:target_end] = 0
            noise_attention = (attention_map * noise_mask).mean().item()
            
            # Store normalized scores for each depth
            attention_analysis['answer_span_scores_per_depth'][depth].append(answer_attention)
            attention_analysis['noise_context_scores_per_depth'][depth].append(noise_attention)
    
    # Calculate average scores per depth
    avg_answer_attention_per_depth = {
        depth: sum(scores) / len(scores) 
        for depth, scores in attention_analysis['answer_span_scores_per_depth'].items()
    }
    avg_noise_attention_per_depth = {
        depth: sum(scores) / len(scores) 
        for depth, scores in attention_analysis['noise_context_scores_per_depth'].items()
    }
    
    attention_ratio_per_depth = {
        depth: avg_answer_attention_per_depth[depth] / avg_noise_attention_per_depth[depth]
        for depth in depths
    }

    results = {
        'avg_answer_attention_per_depth': avg_answer_attention_per_depth,
        'avg_noise_attention_per_depth': avg_noise_attention_per_depth,
        'attention_ratio_per_depth': attention_ratio_per_depth,
        'individual_answer_attention_per_depth': attention_analysis['answer_span_scores_per_depth'],
        'individual_noise_attention_per_depth': attention_analysis['noise_context_scores_per_depth'],
    }

    # Save results to a JSON file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'{folder_name}/target_attention_{timestamp}_epoch_{epoch}.json'
    with open(filename, 'w') as f:
        json.dump(results, f)
    
    return results




def train_model(model, train_dataloader, val_dataloader, num_epochs=20, learning_rate=1e-4, tokenizer=None):
    ''' 
    Training loop for the model.
    '''
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean').to(device)
    
    epochs_seen = []
    tokens_seen = []
    train_losses = []
    val_losses = []
        # Initialize tracking variables
    best_val_loss = float('inf')
    needle_results = []
    persisted_attn_results = {
        'answer_span': {depth: [] for depth in [0.0, 0.25, 0.5, 0.75]},
        'noise_context': {depth: [] for depth in [0.0, 0.25, 0.5, 0.75]}
    }
    # Create results directory
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('attn_maps', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = input_ids[:, 1:].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            batch_size, seq_len, vocab_size = (0,0,0)
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                logits = model(input_ids[:, :-1], mask=attention_mask[:, :-1])
                batch_size, seq_len, vocab_size = logits.shape
                if torch.isnan(logits).any():
                    print(f"NaN detected in logits. Shape: {logits.shape}")
                    continue
                logits = logits.reshape(-1, logits.size(-1))
                target_ids = target_ids.reshape(-1)
                loss = criterion(logits, target_ids)
            # logits = model(input_ids[:, :-1], mask=attention_mask[:, :-1])
            # Reshape logits and targets properly
            # logits = logits.reshape(-1, vocab_size)
            
            # target_ids = target_ids.reshape(-1)
            # loss = criterion(logits, target_ids)
            if torch.isnan(loss):
                print(f"NaN loss detected. Logits min/max: {logits.min():.4f}/{logits.max():.4f}")
            
            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()
            # Step optimizer with scaled gradients
            scaler.step(optimizer)
            scaler.update()
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

            attn_results = analyze_attention_scores(model, tokenizer, epoch, num_samples=50, context_length=seq_len, folder_name='attn_maps')
            for depth in [0.0, 0.25, 0.5, 0.75]:
                persisted_attn_results['answer_span'][depth].append(
                    attn_results['avg_answer_attention_per_depth'][depth]
                )
                persisted_attn_results['noise_context'][depth].append(
                    attn_results['avg_noise_attention_per_depth'][depth]
                )

            plot_attention_analysis(persisted_attn_results, epoch, folder_name='attn_maps')
           
            # Plot progress and generate sample
            plot_training_progress(train_losses, val_losses, epochs_seen, folder_name='plots')
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
            }, 'checkpoints/best_model.pt')
            
        
       
        sample_text = "The quick brown fox"
        generate_and_print_sample(model, tokenizer, device, sample_text)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return model, train_losses, val_losses



if __name__ == "__main__":
    '''
    Starting point for the model.
    Declaring model, optimizer, loss function and training loop.
    '''
    print(f"Using device: {device}")
    print("Loading dataset and tokenizer...")

    tokenizer = setup_tokenizer("gpt2")
    # Prepare DataLoaders
    train_dataloader, val_dataloader = prepare_data(
        dataset_name="wikitext",
        dataset_split="wikitext-2-raw-v1",
        split = "train"
    )

    # Hyperparameter optimization
    try:
         # Initialize model with fixed hyperparameters
        model = DecoderTransformer(
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
            learning_rate=1e-4,
            tokenizer=tokenizer
        )

        torch.save(trained_model.state_dict(), 'final_model.pt')


    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise



