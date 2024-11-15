# from modulefinder import Module
import torch
import torch.nn.functional as F
from torch import nn
# from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
import math
import numpy as np

class PositionEmbedding(nn.Module):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.pos_emb = nn.Embedding(num_embeddings=self.maxlen, embedding_dim=embed_dim)

    def forward(self, x):
        #maxlen = tf.shape(x)[-1]
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
        # tensor x = [batch_size, length, embed_dim]
        # out = [batch_size, head, length, projection_dim]
        length = x.size()[1]
        x = x.view(batch_size, self.num_heads, length, self.projection_dim)
        return x
        # x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        # return tf.transpose(x, perm=[0, 2, 1, 3])
    def concat(self, x):
        batch_size, head, length, d_tensor = x.size()
        d_model = head*d_tensor
        x = x.view(batch_size, length, d_model)
        return x

    def forward(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
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
        # self.activation = _get_activation_fn(activation)

    def forward(self, inputs): #training):
        attention_output = self.self_attn(inputs)
        attention_output = self.dropout1(attention_output) # 1.
        out1 = self.norm1(inputs + attention_output) # 2.
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(out1))))
        # ffn_output = self.linear2(ffn_output)
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
    return torch.Tensor.to(pos_encoding, dtype = torch.float32)
        # return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len, device):
        """
        sin, cos encoding 구현
        
        parameter
        - d_model : model의 차원
        - max_len : 최대 seaquence 길이
        - device : cuda or cpu
        """
        
        super(PositionalEncoding, self).__init__() # nn.Module 초기화
        
        # input matrix(자연어 처리에선 임베딩 벡터)와 같은 size의 tensor 생성
        # 즉, (max_len, d_model) size
        # self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # 인코딩의 그래디언트는 필요 없다. 
        
        # 위치 indexing용 벡터
        # pos는 max_len의 index를 의미한다.
        # pos = torch.arange(0, max_len, device =device)
        pos = torch.arange(0, max_len, device =device)
        # 1D : (max_len, ) size -> 2D : (max_len, 1) size -> word의 위치를 반영하기 위해
        
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32 (없어도 되긴 함)
        
        # i는 d_model의 index를 의미한다. _2i : (d_model, ) size
        # 즉, embedding size가 512일 때, i = [0,512]
        # _2i = torch.arange(0, d_model, step=2, device=device).float()
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # (max_len, 1) / (d_model/2 ) -> (max_len, d_model/2)
        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
        
    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        # batch_size = 128, seq_len = 30
        batch_size, seq_len = x.size() 
        
        # [seq_len = 30, d_model = 512]
        # [128, 30, 512]의 size를 가지는 token embedding에 더해질 것이다. 
        # 
        return self.encoding[:seq_len, :]
    

class TransformerBlock(nn.Module):
    def __init__(self, num_layers, embed_dim, maximum_position_encoding, num_heads=8, ff_dim=2048, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embed_dim)
        # self.pos_encoding = PositionalEncoding(self.embed_dim, maximum_position_encoding, device= 'cuda')
        # print(self.pos_encoding)
        # exit()
        self.enc_layers = [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, rate) 
                        for _ in range(num_layers)]
        self.dropout = nn.Dropout(rate)
    
    def forward(self, x):# training):
        seq_len = x.size()[1]
        # embed_dim = self.embed_dim
        # # print(embed_dim.dtype)
        # embed_dim = torch.FloatTensor(embed_dim)
        # # embed_dim = embed_dim.type(torch.float32)
        # print(embed_dim.dtype)
        # exit()
        # x *= torch.sqrt(embed_dim)
        x *= torch.sqrt(torch.tensor(self.embed_dim, dtype = torch.float32, device='cuda'))
        # x *= torch.sqrt(torch.FloatTensor(self.embed_dim, device='cuda'))
        # adding embedding and position encoding.
        # x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # (batch_size, input_seq_len, d_model)