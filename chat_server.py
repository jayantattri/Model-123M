#!/usr/bin/env python3
"""
Chat server for GPT model with real-time streaming
"""

import os
import json
import time
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass
import threading
import pickle


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


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
        self.transformer.wte.weight = self.lm_head.weight
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
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class ChatGPT:
    def __init__(self, checkpoint_path="model_05000.pt"):
        # Set device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        
        print(f"Loading model on device: {self.device}")
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        
        print(f"Model loaded successfully! (Step: {checkpoint['step']}, Val Loss: {checkpoint['val_loss']:.4f})")
    
    def generate_stream(self, prompt, max_length=150, temperature=0.8, top_k=50):
        """Generate text with streaming"""
        # Encode prompt
        tokens = self.enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate tokens one by one
        with torch.no_grad():
            for i in range(max_length - tokens.size(1)):
                logits, _ = self.model(tokens)
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(-1, top_k_indices, top_k_values)
                    logits = logits_filtered
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # Decode the new token and yield it
                new_token_text = self.enc.decode([next_token.item()])
                yield new_token_text
                
                # Add a small delay for streaming effect
                time.sleep(0.03)


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model instance
chat_model = None
# Store chat sessions with UUID as key
chat_sessions = {}
# Store session metadata
session_metadata = {}

# File paths for persistent storage
STORAGE_DIR = "chat_storage"
CHAT_SESSIONS_FILE = os.path.join(STORAGE_DIR, "chat_sessions.pkl")
SESSION_METADATA_FILE = os.path.join(STORAGE_DIR, "session_metadata.pkl")

# Create storage directory if it doesn't exist
os.makedirs(STORAGE_DIR, exist_ok=True)

def save_chat_data():
    """Save chat sessions and metadata to disk"""
    try:
        with open(CHAT_SESSIONS_FILE, 'wb') as f:
            pickle.dump(chat_sessions, f)
        with open(SESSION_METADATA_FILE, 'wb') as f:
            pickle.dump(session_metadata, f)
        print("Chat data saved successfully!")
    except Exception as e:
        print(f"Error saving chat data: {e}")

def load_chat_data():
    """Load chat sessions and metadata from disk"""
    global chat_sessions, session_metadata
    try:
        if os.path.exists(CHAT_SESSIONS_FILE):
            with open(CHAT_SESSIONS_FILE, 'rb') as f:
                chat_sessions = pickle.load(f)
            print(f"Loaded {len(chat_sessions)} chat sessions")
        
        if os.path.exists(SESSION_METADATA_FILE):
            with open(SESSION_METADATA_FILE, 'rb') as f:
                session_metadata = pickle.load(f)
            print(f"Loaded {len(session_metadata)} session metadata entries")
            
    except Exception as e:
        print(f"Error loading chat data: {e}")
        # If loading fails, start with empty data
        chat_sessions = {}
        session_metadata = {}

def load_model():
    """Load the model on startup"""
    global chat_model
    try:
        chat_model = ChatGPT()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.route('/')
def index():
    """Serve the chat interface"""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with streaming response"""
    if not chat_model:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', '')
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    if not session_id:
        return jsonify({"error": "Session ID required"}), 400
    
    # Initialize session if not exists
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
        session_metadata[session_id] = {
            "created_at": datetime.now().isoformat(),
            "title": user_message[:50] + "..." if len(user_message) > 50 else user_message,
            "last_updated": datetime.now().isoformat()
        }
    
    # Add user message to session history
    user_entry = {
        "type": "user",
        "message": user_message,
        "timestamp": datetime.now().isoformat()
    }
    chat_sessions[session_id].append(user_entry)
    
    def generate():
        """Generator function for streaming response"""
        ai_response = ""
        
        # Send the initial response structure
        yield f"data: {json.dumps({'type': 'start', 'user_message': user_message})}\n\n"
        
        # Generate and stream the AI response
        for token in chat_model.generate_stream(user_message):
            ai_response += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        
        # Add AI response to session history
        ai_entry = {
            "type": "ai",
            "message": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        chat_sessions[session_id].append(ai_entry)
        
        # Update session metadata
        session_metadata[session_id]["last_updated"] = datetime.now().isoformat()
        
        # Save chat data to disk
        save_chat_data()
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete', 'full_response': ai_response})}\n\n"
    
    return Response(generate(), mimetype='text/plain')


@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get chat history for a specific session"""
    if session_id not in chat_sessions:
        return jsonify([])
    return jsonify(chat_sessions[session_id])


@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    sessions = []
    for session_id, metadata in session_metadata.items():
        sessions.append({
            "id": session_id,
            "title": metadata["title"],
            "created_at": metadata["created_at"],
            "last_updated": metadata["last_updated"],
            "message_count": len(chat_sessions.get(session_id, []))
        })
    
    # Sort by last updated (most recent first)
    sessions.sort(key=lambda x: x["last_updated"], reverse=True)
    return jsonify(sessions)


@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = []
    session_metadata[session_id] = {
        "created_at": datetime.now().isoformat(),
        "title": "New Chat",
        "last_updated": datetime.now().isoformat()
    }
    
    # Save chat data to disk
    save_chat_data()
    
    return jsonify({"session_id": session_id})


@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    if session_id in session_metadata:
        del session_metadata[session_id]
    
    # Save chat data to disk
    save_chat_data()
    
    return jsonify({"status": "success"})


if __name__ == '__main__':
    # Load chat data from disk first
    load_chat_data()
    
    # Load model in a separate thread to not block startup
    model_thread = threading.Thread(target=load_model)
    model_thread.start()
    
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    print("Starting chat server on http://localhost:3000")
    print("Make sure to create the templates/chat.html file!")
    
    app.run(host='0.0.0.0', port=3000, debug=True, threaded=True)
