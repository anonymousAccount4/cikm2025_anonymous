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

def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm1d(num_channels)
    elif norm == 'ln':
        return nn.LayerNorm(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")

class EmbedSequential(nn.Sequential):
    def forward(self, x, context=None):
        for layer in self:
            if isinstance(layer, Temporal_Att):
                x = layer(x)
            else:
                x = layer(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.net(x) + x

class AttentionBlock(nn.Module):
  def __init__(self, in_channels, out_channels=None, clip_length=2, n_heads=1):
    super(AttentionBlock, self).__init__()
    self.clip_length = clip_length
    self.n_heads = n_heads
    self.in_channels = in_channels
    if out_channels:
      self.out_channels = out_channels
      self.down = nn.Linear(self.in_channels, self.out_channels)
    else:
       self.out_channels = in_channels
    self.to_qkv = nn.Linear(self.out_channels, self.out_channels*3)
    self.to_out = zero_module(nn.Linear(self.out_channels, self.out_channels))
    self.norm = get_norm('ln', in_channels, 32)

  def forward(self, x):
    x = self.norm(x)
    qkv = self.to_qkv(x)
    bs, length, width = qkv.shape
    ch = width // (3 * self.n_heads)
    q, k, v = torch.split(qkv, self.out_channels, dim=2)
    scale = 1/math.sqrt(math.sqrt(ch))
    weight = torch.einsum(
       "bqc,bkc->bqk",
       (q * scale).view(bs * self.n_heads, length, ch),
       (k * scale).view(bs * self.n_heads, length, ch)
    )
    weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    out = self.to_out(torch.einsum("bqk,bkc->bqc", weight, v.reshape(bs * self.n_heads, length, ch)).reshape(bs,length,-1)) + x
    return out

class Temporal_Att(nn.Module):
  def __init__(self, in_channels):
    super(Temporal_Att, self).__init__()
    self.in_channels = in_channels
    self.att_block = AttentionBlock(in_channels)
    self.ff = FeedForward(in_channels)
  
  def forward(self, x):
    x = x.reshape(-1, 4*3, self.in_channels)
    x = self.att_block(x).reshape(-1, self.in_channels)
    # x = self.ff(x)
    return x

class UpDownsample(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UpDownsample, self).__init__()
    self.down = nn.Linear(in_channels, out_channels)

  def forward(self, x):
    return self.down(x)

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels=None, dropout=0.0):
    super(ResBlock, self).__init__()
    self.in_channels = in_channels
    self.dropout = dropout
    if in_channels != out_channels:
      self.updown = True
      self.out_channels = out_channels
      self.x_upd = UpDownsample(in_channels, out_channels)
    else:
      self.updown = False
      self.out_channels = in_channels

    self.in_layers = nn.Sequential(
      nn.BatchNorm1d(in_channels),
      nn.GELU(),
      nn.Linear(in_channels, out_channels)
    )
    self.out_layers = nn.Sequential(
      nn.BatchNorm1d(out_channels),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(out_channels, out_channels)
    )

  def forward(self, x):
    h = self.in_layers(x)
    h = self.out_layers(h)
    if self.updown:
      x = self.x_upd(x)
    return h + x

class AutoEncoder(nn.Module):
  def __init__(self, in_channel, ch_mult=[0.5,2,2]):
    super(AutoEncoder, self).__init__()
    self.up_sample = nn.Linear(in_channel, in_channel*3)
    layers = []
    for level, mult in enumerate(ch_mult):
      layers.append(Temporal_Att(in_channel))
      layers.append(nn.LayerNorm(in_channel))
      layers.append(nn.GELU())
      layers.append(nn.Linear(in_channel, int(in_channel//mult)))
      in_channel = int(in_channel//mult)
    self.encoder = EmbedSequential(*layers)

    layers = []
    for level, mult in enumerate(ch_mult[::-1]):
      layers.append(Temporal_Att(in_channel))
      layers.append(nn.LayerNorm(in_channel))
      layers.append(nn.GELU())
      layers.append(nn.Linear(in_channel, int(in_channel*mult)))
      in_channel = int(in_channel*mult)
    self.decoder = EmbedSequential(*layers)

  def forward(self, x):
    B, C = x.shape
    x = self.up_sample(x.reshape(-1, C)).reshape(-1, C)
    h = self.encoder(x)
    hat_pst, hat_prst, hat_ftr = self.decoder(h).reshape(-1, 4, 3, C).permute(2, 0, 1, 3)
    return hat_pst.reshape(-1, C), hat_prst.reshape(-1, C), hat_ftr.reshape(-1, C)