import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=32, latent_size=8, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_enc = nn.Linear(hidden_size, latent_size)
        self.fc_dec = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)
    

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        z = self.fc_enc(enc_out[:, -1, :])
        dec_input = self.fc_dec(z).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(dec_input)
        return out
