import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoders = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.encoders.append(self.conv_block(in_channels if i == 0 else hidden_dims[i - 1], hidden_dims[i]))

        self.decoders = nn.ModuleList()
        hidden_dims = hidden_dims[::-1]
        for i in range(len(hidden_dims)-1):
            self.decoders.append(self.conv_block(hidden_dims[i], (hidden_dims[i + 1])))
        self.final = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        encoded_outs = []
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            if i != len(self.encoders) - 1:
                encoded_outs.append(x)
                x = self.pool(x)
        
        encoded_outs = encoded_outs[::-1]
        x = self.up(x)
        for i in range(len(encoded_outs)):
            x = torch.cat((x, encoded_outs[i]), dim=-1)
            x = self.decoders[i](x)
            x = self.up(x)
        x = self.final(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, image_dim, timesteps=1000, beta_start=0.1, beta_end=0.2):
        super(DiffusionModel, self).__init__()
        self.image_dim = image_dim
        self.timesteps = timesteps

        # Linear schedule for beta
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        # Precompute alpha and their cumulative products
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]])

        self.unet = UNet(1, 1).to(device)

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0).to(device)
        alpha_t = self.alpha_cumprod[t][:, None, None, None]
        return torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise, noise

    def reverse_diffusion(self, x_t, t):
        alpha_t = self.alpha_cumprod[t][:, None, None, None]
        beta_t = self.betas[t][:, None, None, None]
        predicted_noise = self.unet(x_t)
        return (x_t - beta_t * predicted_noise) / torch.sqrt(alpha_t)

    def loss_function(self, x_0, t, noise):
        x_t, true_noise = self.forward_diffusion(x_0, t)
        predicted_noise = self.unet(x_t)
        return F.mse_loss(predicted_noise, true_noise)
