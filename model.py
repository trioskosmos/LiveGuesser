import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LoveLiveTransformer(nn.Module):
    def __init__(self, num_songs, num_artists, num_feedback_types, num_lives, d_model=64, nhead=4, num_layers=2):
        super(LoveLiveTransformer, self).__init__()

        # Use padding_idx=0 so embedding for 0 is always vector of zeros
        self.song_embedding = nn.Embedding(num_songs, d_model, padding_idx=0)
        self.artist_embedding = nn.Embedding(num_artists, d_model, padding_idx=0)
        self.feedback_embedding = nn.Embedding(num_feedback_types, d_model, padding_idx=0)

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc_out = nn.Linear(d_model, num_lives)
        self.d_model = d_model

    def forward(self, song_seq, artist_seq, feedback_seq):
        # seq input: (seq_len, batch_size)

        # Create padding mask (batch_size, seq_len)
        # True where value is 0 (padding)
        src_key_padding_mask = (song_seq == 0).transpose(0, 1)

        # Embed inputs
        src = self.song_embedding(song_seq) + self.artist_embedding(artist_seq) + self.feedback_embedding(feedback_seq)

        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Transformer Encoder
        # output: (seq_len, batch_size, d_model)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Pooling: Mean pooling excluding padding
        # Transpose to (batch_size, seq_len, d_model)
        output = output.transpose(0, 1)

        # Create mask for pooling (batch_size, seq_len, 1)
        # 1 for valid, 0 for padding
        mask = (~src_key_padding_mask).float().unsqueeze(2)

        # Sum valid outputs
        sum_output = torch.sum(output * mask, dim=1)

        # Count valid tokens
        count_valid = torch.sum(mask, dim=1)

        # Avoid division by zero
        count_valid = torch.clamp(count_valid, min=1.0)

        # Mean
        pooled_output = sum_output / count_valid

        # Classification
        logits = self.fc_out(pooled_output)
        return logits
