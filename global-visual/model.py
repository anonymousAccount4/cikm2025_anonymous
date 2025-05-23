import torch.nn as nn
import torch
import torch.nn.functional as F
import math

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class AutoEncoder(nn.Module):
    def __init__(self, in_channel, ch_mult=[0.5,2,2,2]):
        super(AutoEncoder, self).__init__()
        self.up_sample = nn.Linear(in_channel, in_channel*3)
        layers = []
        for level, mult in enumerate(ch_mult):
            layers.append(nn.LayerNorm(in_channel))
            layers.append(nn.GELU())
            layers.append(nn.Linear(in_channel, int(in_channel//mult)))
            in_channel = int(in_channel//mult)
        self.encoder = nn.Sequential(*layers)

        layers = []
        for level, mult in enumerate(ch_mult[::-1]):
            layers.append(nn.LayerNorm(in_channel))
            layers.append(nn.GELU())
            layers.append(nn.Linear(in_channel, int(in_channel*mult)))
            in_channel = int(in_channel*mult)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        B, C = x.shape 
        x = self.up_sample(x.reshape(-1, C)).reshape(-1, C)
        x = self.encoder(x)
        hat_pst, hat_prst, hat_ftr = self.decoder(x).reshape(-1, 3, C).permute(1, 0, 2)
        return hat_pst, hat_prst, hat_ftr
    