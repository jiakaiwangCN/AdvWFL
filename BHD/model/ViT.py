import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.MixtureNorm import MixtureNorm1d


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        if len(x.shape) == 2:
            batch_size, dim = x.shape
            seq_len = 1
        elif len(x.shape) == 3:
            batch_size, seq_len, dim = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_weights.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.fc(out)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim),
        )
    def forward(self, x):
        attn_out = self.self_attn(x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, dim, num_heads, mlp_dim, input_dim, output_dim):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(dim, output_dim)
        self.fc2 = nn.Linear(dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)  # Global average pooling
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2

class ViT(nn.Module):
    def __init__(self, input_dim):
        super(ViT, self).__init__()
        self.vit = TransformerEncoder(num_layers=6, dim=64, num_heads=8, mlp_dim=128, input_dim=input_dim, output_dim=1)
    
    def forward(self, x, data_type='normal'):
        return self.vit(x)