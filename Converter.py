import torch
import torch.nn as nn
import math
import numpy as np

class Converter(nn.Module):
    def __init__(self, max_seq_len, d_model=64, nhead=8, num_layers=6, dim_feedforward=256, dropout=0.1):
        super(Converter, self).__init__()
        
        self.d_model = d_model
        
        self.input_embedding = nn.Linear(1, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.output_linear = nn.Linear(d_model, 20)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, src_key_padding_mask=None):
        # x shape: (seq_len, batch_size, 1)
        
        x = self.input_embedding(x)  # Now: (seq_len, batch_size, d_model)
        
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        x = self.output_linear(x)  # Now: (seq_len, batch_size, 20)
        x = self.softmax(x)
        
        # Convert softmaxxed matrices into one-dimensional indeces
        with torch.no_grad():
            out = []
            for i in range(len(x)):
                out.append([])
                for j in range(len(x[i])):
                    out[-1].append((torch.argmax(x[i][j].detach().cpu())).item()) 
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def create_padding_mask(sequences, pad_value=0):
    # sequences shape: (seq_len, batch_size, 1)
    return (sequences.squeeze(-1) == pad_value).t()  # (batch_size, seq_len)
