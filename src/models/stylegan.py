import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Helper Modules ---

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(w_dim, channels)
        self.style_bias = nn.Linear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight * noise

# --- Generator ---

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, hidden_dim=256, n_layers=8):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_layers):
            in_dim = z_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, w_dim))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.noise1 = NoiseInjection(out_channels)
        self.noise2 = NoiseInjection(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.lrelu(x)
        x = self.adain1(x, w)
        
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.lrelu(x)
        x = self.adain2(x, w)
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, out_channels=3, img_size=128):
        super().__init__()
        self.w_dim = w_dim
        self.mapping_network = MappingNetwork(z_dim, w_dim)
        
        self.start_const = nn.Parameter(torch.ones(1, 512, 4, 4))
        
        self.blocks = nn.ModuleList([
            GenBlock(512, 512, w_dim),   # 4x4
            GenBlock(512, 512, w_dim),   # 8x8
            GenBlock(512, 256, w_dim),   # 16x16
            GenBlock(256, 128, w_dim),   # 32x32
            GenBlock(128, 64, w_dim),    # 64x64
            GenBlock(64, 32, w_dim),     # 128x128
        ])
        
        self.to_rgb = nn.Conv2d(32, out_channels, 1)

    def forward(self, z, return_w=False):
        w = self.mapping_network(z)
        x = self.start_const.repeat(z.shape[0], 1, 1, 1)
        
        for i, block in enumerate(self.blocks):
            if i > 0:
                 x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = block(x, w)
            
        img = torch.tanh(self.to_rgb(x))
        
        if return_w:
            return img, w
        return img

# --- Discriminator ---

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, img_size=64):
        super().__init__()
        
        self.from_rgb = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.LeakyReLU(0.2),
        )
        
        self.blocks = nn.ModuleList([
            ConvBlock(64, 128),     # 64 -> 32
            ConvBlock(128, 256),    # 32 -> 16
            ConvBlock(256, 512),    # 16 -> 8
            ConvBlock(512, 512),    # 8 -> 4
        ])
        
        self.final_block = nn.Sequential(
            nn.Conv2d(512, 512, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 1, 1, 0)
        )

    def forward(self, x):
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
            x = F.avg_pool2d(x, 2)
        x = self.final_block(x)
        return x.view(x.shape[0], -1)


if __name__ == '__main__':
    # --- Device Configuration ---
    def get_device():
        """Checks for CUDA, DirectML, and falls back to CPU."""
        if torch.cuda.is_available(): return torch.device("cuda")
        try:
            import torch_directml
            if torch_directml.is_available(): return torch_directml.device()
        except (ImportError, AttributeError): pass
        return torch.device("cpu")
    DEVICE = get_device()
    print(f"Running StyleGAN test on device: {DEVICE}")
    # --- End Device Configuration ---

    # A simple test to verify the StyleGAN models
    Z_DIM = 128
    W_DIM = 128
    IMG_SIZE = 64 # Test with 64
    
    # Need to use the IntegratedGenerator for the test now
    from .integrated import IntegratedGenerator
    
    gen = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
    disc = Discriminator(img_size=IMG_SIZE).to(DEVICE)
    
    # Test Generator
    z = torch.randn(2, Z_DIM).to(DEVICE)
    img, w = gen(z, return_w=True)
    print(f"Generator test:")
    print(f"  Input z shape: {z.shape}")
    print(f"  Output image shape: {img.shape}")
    print(f"  Output w shape: {w.shape}")
    assert img.shape == (2, 3, IMG_SIZE, IMG_SIZE)
    
    # Test Discriminator
    output = disc(img)
    print(f"\nDiscriminator test:")
    print(f"  Input image shape: {img.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 1)

    print("\nStyleGAN models created successfully!") 