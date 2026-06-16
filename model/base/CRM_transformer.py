import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()

        # Project to q, k, v
        q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, embed_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, embed_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                    embed_dim)  # (batch_size, seq_len, embed_dim)

        # Final linear projection
        output = self.out_proj(attn_output)  # (batch_size, seq_len, embed_dim)

        return output, attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=2048):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        # Self-attention
        attn_output, attn_weights = self.attention(x)
        x = self.norm1(x + attn_output)  # Residual connection and layer norm

        # Feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)  # Residual connection and layer norm

        return x, attn_weights


class TransformerCRM(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=2, output_channel=128):
        super(TransformerCRM, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Projection to embedding dimension
        self.projection = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Output projection
        self.output_conv1 = nn.Conv2d(embed_dim, output_channel * 2, kernel_size=3, padding=1)
        self.output_conv2 = nn.Conv2d(output_channel * 2, output_channel, kernel_size=3, padding=1)

    def forward(self, feature):
        # Store original shape
        original_shape = feature.shape
        batch_size, channels, height, width = original_shape

        # Project to embedding dimension if needed
        if channels != self.embed_dim:
            feature = self.projection(feature)
            channels = self.embed_dim

        # Reshape feature map to sequence: (batch, channels, height, width) -> (batch, seq_len, channels)
        feature_flat = feature.view(batch_size, channels, -1).transpose(1, 2)  # (batch, seq_len, channels)

        # Apply transformer blocks
        attention_weights = []
        x = feature_flat
        for transformer in self.transformer_blocks:
            x, attn_weights = transformer(x)
            attention_weights.append(attn_weights)

        # Reshape back to feature map: (batch, seq_len, channels) -> (batch, channels, height, width)
        x = x.transpose(1, 2).view(batch_size, channels, height, width)

        # Apply output convolutions
        x1 = F.relu(self.output_conv1(x))
        x2 = F.relu(self.output_conv2(x1))

        # Concatenate original feature with transformed features
        output = torch.cat([feature, x1, x2], dim=1)

        return output
