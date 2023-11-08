import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
#from torchtext.vocab import vocab
#import torchtext.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import math
#import janome
#from janome.tokenizer import Tokenizer
#import spacy
#from collections import Counter


class PositionalEncoding(nn.Module):
    def __init__(self, dim, batch_size, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_size = batch_size
        #position = torch.arange(max_len).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        #pe = torch.zeros(max_len, 1, dim)
        #pe[:, 0, 0::2] = torch.sin(position * div_term)
        #pe[:, 0, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #x = x + self.pe[:x.size(0)]
        #self.pe = self.pe.expand(self.batch_size,-1, -1)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head_num, dropout = 0.1):
        super().__init__() 
        self.dim = dim
        self.head_num = head_num
        self.linear_Q = nn.Linear(dim, dim, bias = False)
        self.linear_K = nn.Linear(dim, dim, bias = False)
        self.linear_V = nn.Linear(dim, dim, bias = False)
        self.linear = nn.Linear(dim, dim, bias = False)
        self.soft = nn.Softmax(dim = 3)
        self.dropout = nn.Dropout(dropout)
        
    def split_head(self, x):
        split_size = x.size(2) // self.head_num
        x = torch.split(x, split_size, dim=2)
        
        x = torch.stack(x, dim = 1)
        return x
    
    def concat_head(self, x):
        x_splits = x.split(1, dim=1)
        x = torch.cat(x_splits, dim=3)  # 4次元目に沿ってテンソルを連結します。
        x = x.squeeze(dim=1)
        return x
    
    def forward(self, Q, K, V, mask = None):
        Q = self.linear_Q(Q)   #(BATCH_SIZE,word_count,dim)
        K = self.linear_K(K)
        V = self.linear_V(V)
    
        Q = self.split_head(Q)   #(BATCH_SIZE,head_num,word_count//head_num,dim)
        K = self.split_head(K)
        V = self.split_head(V)

        QK = torch.matmul(Q, torch.transpose(K, 3, 2))
        QK = QK/((self.dim//self.head_num)**0.5)

        if mask is not None:
            QK = QK + mask

        softmax_QK = self.soft(QK)
        softmax_QK = self.dropout(softmax_QK)

        QKV = torch.matmul(softmax_QK, V)
        QKV = self.concat_head(QKV)
        QKV = self.linear(QKV)
        return QKV

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim = 3072, dropout = 0.1):
        super().__init__() 
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim, head_num, dropout = 0.1):
        super().__init__() 
        self.MMHA = MultiHeadAttention(dim, head_num)
        #self.MHA = MultiHeadAttention(dim, head_num)
        self.layer_norm_1 = nn.LayerNorm([dim])
        self.layer_norm_2 = nn.LayerNorm([dim])
        self.layer_norm_3 = nn.LayerNorm([dim])
        self.FF = FeedForward(dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        Q = K = V = x
        x = self.MMHA(Q, K, V, mask)
        x = self.dropout_1(x)
        x = x + Q
        x = self.layer_norm_1(x)
        #Q = x
        #K = V = y
        #x = self.MHA(Q, K, V)
        #x = self.dropout_2(x)
        #x = x + Q
        #x = self.layer_norm_2(x)
        _x = x
        x = self.FF(x)
        x = self.dropout_3(x)
        x = x + _x
        x = self.layer_norm_3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, tag_vocab_size, dim, head_num, batch_size, dropout = 0.1):
        super().__init__() 
        self.dim = dim
        self.embed = nn.Embedding(dec_vocab_size + tag_vocab_size + 1 , dim, padding_idx=0)
        self.PE = PositionalEncoding(dim, batch_size)
        self.DecoderBlocks = nn.ModuleList([DecoderBlock(dim, head_num) for _ in range(12)])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, dec_vocab_size)
        
    def forward(self, x, mask):
        x = x.long()
        x = self.embed(x)
        x = x*(self.dim**0.5)
        x = self.PE(x)
        x = self.dropout(x)
        for i in range(12):
            x = self.DecoderBlocks[i](x, mask)
        x = self.linear(x) #損失の計算にnn.CrossEntropyLoss()を使用する為、Softmax層を挿入しない
        return x

class Decoder_only_Transformer(nn.Module):
    def __init__(self, dec_vocab_size, tag_vocab_size, dim, head_num, batch_size):
        super().__init__() 
        self.decoder = Decoder(dec_vocab_size, tag_vocab_size, dim, head_num, batch_size)
        
    def forward(self, dec_input, mask):
        output = self.decoder(dec_input, mask)
        return output
