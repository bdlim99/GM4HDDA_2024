import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def load_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.load_state_dict(ckpt['model_state'])
