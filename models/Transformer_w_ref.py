import torch
from torch import nn

from remora import constants


class network(nn.Module):
    _variable_width_possible = False

    def __init__(
        self,
        size=constants.DEFAULT_NN_SIZE,
        kmer_len=constants.DEFAULT_KMER_LEN,
        num_out=2,
        num_layers=2,
        num_heads=4,
        ff_mult=4,
        dropout=0.1,
    ):
        super().__init__()
        if size % 2 != 0:
            raise ValueError("Transformer model size must be even")
        if size % num_heads != 0:
            raise ValueError("Transformer model size must be divisible by heads")

        half = size // 2
        self.sig_proj = nn.Conv1d(1, half, kernel_size=1)
        self.seq_proj = nn.Conv1d(kmer_len * 4, half, kernel_size=1)
        self.norm_in = nn.LayerNorm(size)

        self.max_positions = 4096
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_positions, size) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=size,
            nhead=num_heads,
            dim_feedforward=size * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(size)
        self.fc = nn.Linear(size, num_out)

    def forward(self, sigs, seqs):
        sig_features = self.sig_proj(sigs)
        seq_features = self.seq_proj(seqs)
        x = torch.cat((sig_features, seq_features), dim=1).transpose(1, 2)
        if x.size(1) > self.max_positions:
            raise ValueError(
                f"Chunk length {x.size(1)} exceeds max_positions "
                f"{self.max_positions}"
            )
        pos = self.pos_embedding[:, : x.size(1), :]
        x = self.norm_in(x + pos)
        x = self.encoder(x)
        x = self.out_norm(x)
        x = x.mean(dim=1)
        return self.fc(x)
