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
        
        # Perception network with LeakyReLU and BatchNorm
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
        
        self.perceive = nn.Conv2d(self.channel_n, self.channel_n * 2, 3, padding=1, bias=False, groups=self.channel_n)
        self.perceive.weight = nn.Parameter(torch.cat([sobel_x, sobel_y], dim=0), requires_grad=False)
        
        # Enhanced perception processing
        self.perception_net = nn.Sequential(
            nn.Conv2d(self.channel_n * 3, hidden_n, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(hidden_n),
            nn.Conv2d(hidden_n, hidden_n, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(hidden_n)
        )
        
        # Update network with LeakyReLU and BatchNorm
        self.update_net = nn.Sequential(
            nn.Linear(self.channel_n * 3 + w_dim, hidden_n),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_n),
            nn.Linear(hidden_n, self.channel_n),
            nn.Tanh()  # Keep Tanh for final output to maintain bounded values
        )
        
        # Initialize last layer with small weights for stability
        self.update_net[-2].weight.data.fill_(0.0)
        
        # Gradient scaling factors for controlled updates
        self.perception_scale = nn.Parameter(torch.ones(1))
        self.update_scale = nn.Parameter(torch.ones(1))
        
        # Adaptive stability parameters
        self.stability_controller = nn.Sequential(
            nn.Linear(self.channel_n + w_dim, hidden_n // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_n // 2, 3),  # 3 parameters: update_rate, survival_threshold, growth_rate
            nn.Sigmoid()  # Keep parameters in [0,1] range
        )

    def get_stability_params(self, x, w):
        """Get adaptive stability parameters based on current state"""
        # Global state vector
        global_state = x.mean(dim=[2, 3])  # [batch, channels]
        
        # Input to stability controller
        stability_input = torch.cat([global_state, w], dim=1)
        
        # Get parameters [batch, 3]
        params = self.stability_controller(stability_input)
        
        # Scale parameters to appropriate ranges
        update_rate = params[:, 0:1] * 0.2 + 0.1  # 0.1 to 0.3
        survival_threshold = params[:, 1:2] * 0.1 + 0.1  # 0.1 to 0.2
        growth_rate = params[:, 2:3] * 0.3 + 0.2  # 0.2 to 0.5
        
        return {
            'update_rate': update_rate,
            'survival_threshold': survival_threshold,
            'growth_rate': growth_rate
        }

    def get_living_mask(self, x):
        """Enhanced living mask with adaptive thresholding"""
        alpha = x[:, 3:4, :, :]
        neighbor_count = F.max_pool2d((alpha > 0.1).float(), kernel_size=3, stride=1, padding=1)
        return (alpha > 0.1) & (neighbor_count >= 2)  # Need at least 2 living neighbors

    def forward(self, x, w, steps=64):
        """Forward pass with enhanced stability mechanisms"""
        w_expanded = w.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        
        for _ in range(steps):
            # Get adaptive stability parameters
            stability_params = self.get_stability_params(x, w)
            
            # Pre-update life mask
            pre_life_mask = self.get_living_mask(x)
            
            # Perception with gradient scaling
            perception = self.perceive(x)
            perception = perception * self.perception_scale
            
            # Process perception features
            perception_features = self.perception_net(torch.cat([x, perception], dim=1))
            
            # Prepare update input
            combined_input = torch.cat([x, perception_features, w_expanded], dim=1).permute(0, 2, 3, 1)
            
            # Generate update with scaled gradients
            delta = self.update_net(combined_input).permute(0, 3, 1, 2)
            delta = delta * self.update_scale
            
            # Apply update with adaptive rate
            update_rate = stability_params['update_rate'].view(-1, 1, 1, 1)
            x = x + delta * update_rate
            
            # Apply life dynamics with adaptive parameters
            survival_threshold = stability_params['survival_threshold'].view(-1, 1, 1, 1)
            growth_rate = stability_params['growth_rate'].view(-1, 1, 1, 1)
            
            # Enhanced life dynamics
            neighbor_life = F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1)
            life_mask = (neighbor_life > survival_threshold).float()
            
            # Apply growth factor
            growth_factor = torch.sigmoid(neighbor_life * growth_rate)
            x = x * growth_factor
            
            # Add minimal noise during training
            if self.training:
                x = x + torch.randn_like(x) * 0.001
            
            # Ensure bounded values
            x = x.clamp(-1.0, 1.0)
        
        return x

    def to_rgba(self, x):
        """Convert output to RGBA format"""
        return x[:, :4, :, :].clamp(-1.0, 1.0)
    
    def get_seed(self, batch_size=1, size=128, device='cpu'):
        """Create a seed pattern"""
        seed = torch.zeros(batch_size, self.channel_n, size, size, device=device)
        seed[:, 3:, size // 2-1:size // 2+1, size // 2-1:size // 2+1] = 1.0
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