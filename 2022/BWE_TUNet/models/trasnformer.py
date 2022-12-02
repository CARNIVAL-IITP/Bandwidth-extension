import torch

import math
import numpy as np

from torch import nn
import torch.nn.functional as F


class PositionEmbedding(nn.Module):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.pos_emb = nn.Embedding(num_embeddings=self.maxlen, embedding_dim=embed_dim)

    def forward(self, x):
        positions = torch.range(start=0, end=self.maxlen, step=1)
        positions = self.pos_emb(positions)


class ScaleDotProuctAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProuctAttention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, query, key, value):
        batch_size, head, length, d_tensor = key.size()
        k_t = key.view(batch_size, head, d_tensor, length)
        score = (query @ k_t) / math.sqrt(d_tensor)

        weights = self.softmax(score)
        output = weights @ value
        return output, weights

        
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = nn.Linear(embed_dim, embed_dim, device='cuda')
        self.key_dense = nn.Linear(embed_dim, embed_dim, device='cuda')
        self.value_dense = nn.Linear(embed_dim, embed_dim, device='cuda')
        self.combine_heads = nn.Linear(embed_dim, embed_dim, device='cuda')
        self.attention = ScaleDotProuctAttention() 

    def separate_heads(self, x, batch_size):
        length = x.size()[1]
        x = x.view(batch_size, self.num_heads, length, self.projection_dim)
        return x

    def concat(self, x):
        batch_size, head, length, d_tensor = x.size()
        d_model = head*d_tensor
        x = x.view(batch_size, length, d_model)
        return x

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        concat_attention = self.concat(attention)
           # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  
            # (batch_size, seq_len, embed_dim)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, device='cuda')
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device='cuda')

        self.norm1 = nn.LayerNorm(d_model, device='cuda')
        self.norm2 = nn.LayerNorm(d_model, device='cuda')
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, inputs): #training):
        attention_output = self.self_attn(inputs)
        attention_output = self.dropout1(attention_output) # 1.
        out1 = self.norm1(inputs + attention_output) # 2.
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(out1))))
        src = out1 + self.dropout2(ffn_output) # 4-1.
        return self.norm2(src) # 4-2.
        # return src
    
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
    
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    # return torch.from_numpy(pos_encoding, dtype = torch.float32)
    return torch.tensor(pos_encoding, dtype = torch.float32, device='cuda')


class TransformerBlock(nn.Module):
    def __init__(self, num_layers, embed_dim, maximum_position_encoding, num_heads=8, ff_dim=2048, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embed_dim)

        self.enc_layers = [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, rate) 
                        for _ in range(num_layers)]
        self.dropout = nn.Dropout(rate)
    
    def forward(self, x):
        seq_len = x.size()[1]

        x *= torch.sqrt(torch.tensor(self.embed_dim, dtype = torch.float32, device='cuda'))

        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # (batch_size, input_seq_len, d_model)