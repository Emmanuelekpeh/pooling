import torch
import torch.nn as nn
import torch.nn.functional as F

# Import building blocks from the original StyleGAN
from .stylegan import PixelNorm, AdaIN, NoiseInjection, MappingNetwork

# 1. The Shared Core Module
class SharedBrain(nn.Module):
    """A simple conv network that will be shared between the Generator and the NCA."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
    def forward(self, x):
        return self.conv(x)

# 2. The Integrated StyleGAN Generator
class IntegratedGenBlock(nn.Module):
    """A Generator block that uses the SharedBrain."""
    def __init__(self, in_channels, out_channels, w_dim, shared_brain):
        super().__init__()
        self.shared_brain = shared_brain
        # Other layers remain specific to the generator
        self.noise1 = NoiseInjection(out_channels)
        self.noise2 = NoiseInjection(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)
        self.lrelu = nn.LeakyReLU(0.2)
        # Upsampling is handled outside the block
        self.conv_non_shared = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, w):
        x = self.shared_brain(x) # Pass through the shared part
        x = self.noise1(x)
        x = self.lrelu(x)
        x = self.adain1(x, w)
        
        x = self.conv_non_shared(x) # A non-shared conv layer
        x = self.noise2(x)
        x = self.lrelu(x)
        x = self.adain2(x, w)
        return x

class IntegratedGenerator(nn.Module):
    def __init__(self, z_dim, w_dim, out_channels=3):
        super().__init__()
        self.w_dim = w_dim
        self.mapping_network = MappingNetwork(z_dim, w_dim)
        
        self.start_const = nn.Parameter(torch.ones(1, 512, 4, 4))
        
        # Define channel progression for 64x64 output
        channels = [512, 512, 256, 128, 64] # Removed last stage for 64x64
        
        # Create shared brains and generator blocks
        self.shared_brains = nn.ModuleList()
        self.gen_blocks = nn.ModuleList()
        
        # Initial block is not shared
        self.initial_block = IntegratedGenBlock(512, 512, w_dim, SharedBrain(512, 512))

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i+1]
            shared_brain = SharedBrain(in_ch, out_ch)
            self.shared_brains.append(shared_brain)
            self.gen_blocks.append(IntegratedGenBlock(in_ch, out_ch, w_dim, shared_brain))
            
        self.to_rgb = nn.Conv2d(channels[-1], out_channels, 1)

    def forward(self, z, return_w=False):
        w = self.mapping_network(z)
        x = self.start_const.repeat(z.shape[0], 1, 1, 1)
        x = self.initial_block(x, w)
        
        for block in self.gen_blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = block(x, w)
            
        img = torch.tanh(self.to_rgb(x))
        
        return (img, w) if return_w else img

# 3. The Integrated NCA Model
class IntegratedNCA(nn.Module):
    def __init__(self, channel_n=16, hidden_n=128, w_dim=128):
        super().__init__()
        self.channel_n = channel_n
        
        # Perception network (as before)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
        self.perceive = nn.Conv2d(self.channel_n, self.channel_n * 2, 3, padding=1, bias=False, groups=self.channel_n)
        self.perceive.weight = nn.Parameter(torch.cat([sobel_x, sobel_y], dim=0), requires_grad=False)
        
        # Update network now includes the w_dim
        self.update_net = nn.Sequential(
            nn.Linear(self.channel_n * 3 + w_dim, hidden_n),
            nn.ReLU(),
            nn.Linear(hidden_n, self.channel_n, bias=False)
        )
        self.update_net[-1].weight.data.fill_(0.0)

    def get_living_mask(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def forward(self, x, w, steps=64):
        w_expanded = w.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        
        for _ in range(steps):
            pre_life_mask = self.get_living_mask(x)
            
            perception = self.perceive(x)
            
            combined_input = torch.cat([x, perception, w_expanded], dim=1).permute(0, 2, 3, 1)
            
            delta = self.update_net(combined_input).permute(0, 3, 1, 2)
            
            # Stochastic update
            update_mask = (torch.rand_like(x[:, 3:4, :, :]) < 0.5) & (x[:, 3:4, :, :] > 0.1)
            x = x + delta * update_mask.float()
            
            # Death mask
            post_life_mask = self.get_living_mask(x)
            life_mask = pre_life_mask & post_life_mask
            x = x * life_mask.float()
            
        return x

    def to_rgba(self, x):
        return x[:, :4, :, :].clamp(-1.0, 1.0) # Clamp to valid range for tanh
    
    def get_seed(self, batch_size=1, size=128, device='cpu'):
        seed = torch.zeros(batch_size, self.channel_n, size, size, device=device)
        seed[:, 3:, size // 2, size // 2] = 1.0 
        return seed

if __name__ == '__main__':
    # A simple test to verify the integrated models
    Z_DIM, W_DIM, IMG_SIZE = 128, 128, 128
    
    # Test Generator
    print("--- Testing IntegratedGenerator ---")
    gen = IntegratedGenerator(Z_DIM, W_DIM)
    z = torch.randn(2, Z_DIM)
    img, w = gen(z, return_w=True)
    print(f"  Input z shape: {z.shape}")
    print(f"  Output image shape: {img.shape}")
    print(f"  Output w shape: {w.shape}")
    assert img.shape == (2, 3, IMG_SIZE, IMG_SIZE)
    
    # Test NCA
    print("\n--- Testing IntegratedNCA ---")
    nca = IntegratedNCA(w_dim=W_DIM)
    seed = nca.get_seed(batch_size=2, size=IMG_SIZE)
    nca_grid = nca(seed, w, steps=10)
    nca_img = nca.to_rgba(nca_grid)
    print(f"  Input seed shape: {seed.shape}")
    print(f"  Input w shape: {w.shape}")
    print(f"  Output grid shape: {nca_grid.shape}")
    print(f"  Output image shape: {nca_img.shape}")
    assert nca_img.shape == (2, 4, IMG_SIZE, IMG_SIZE)

    print("\nIntegrated models created successfully!") 