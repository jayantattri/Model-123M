import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 512 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def load_model(checkpoint_path, device='cpu'):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get the config and create model
    config = checkpoint['config']
    model = GPT(config)
    
    # Load the state dict
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training step: {checkpoint['step']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Model config: {config}")
    
    return model, config


def generate_text(model, prompt, max_length=100, temperature=1.0, top_k=50, num_samples=1, device='cpu'):
    """Generate text from the model given a prompt"""
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode the prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1)  # (num_samples, prompt_length)
    
    print(f"Prompt: '{prompt}'")
    print(f"Generating {num_samples} sample(s) with max_length={max_length}, temperature={temperature}, top_k={top_k}")
    print("-" * 80)
    
    model.eval()
    with torch.no_grad():
        # Generate tokens
        for _ in range(max_length - tokens.size(1)):
            # Forward pass
            logits, _ = model(tokens)  # (num_samples, seq_len, vocab_size)
            
            # Take logits from the last position
            logits = logits[:, -1, :]  # (num_samples, vocab_size)
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    # Get top-k values and indices
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    
                    # Set all non-top-k values to very negative (essentially -inf)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(-1, top_k_indices, top_k_values)
                    logits = logits_filtered
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # (num_samples, 1)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)  # (num_samples, 1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
    
    # Decode and return results
    results = []
    for i in range(num_samples):
        token_list = tokens[i].tolist()
        generated_text = enc.decode(token_list)
        results.append(generated_text)
        
        print(f"Sample {i + 1}:")
        print(f"{generated_text}")
        print("-" * 80)
    
    return results


def main():
    """Main function for interactive text generation"""
    
    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = "model_05000.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} not found!")
        print("Make sure you have a trained model checkpoint in the current directory.")
        return
    
    model, config = load_model(checkpoint_path, device)
    
    print("\n" + "="*80)
    print("GPT Text Generator")
    print("="*80)
    print("Enter your prompts below. Type 'quit' to exit.")
    print("You can also use these example prompts:")
    print("- 'The future of artificial intelligence'")
    print("- 'Once upon a time in a distant galaxy'")
    print("- 'The key to happiness is'")
    print("="*80)
    
    while True:
        try:
            # Get user input
            prompt = input("\nEnter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                print("Please enter a non-empty prompt.")
                continue
            
            # Generation parameters (you can modify these)
            max_length = 80
            temperature = 0.8
            top_k = 50
            num_samples = 2
            
            print(f"\nGenerating text...")
            
            # Generate text
            generate_text(
                model=model,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                num_samples=num_samples,
                device=device
            )
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
