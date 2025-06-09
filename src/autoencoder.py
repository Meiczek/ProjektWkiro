import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_size, hidden_size=128, latent_size=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
