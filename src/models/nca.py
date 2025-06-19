import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NCAModel(nn.Module):
    def __init__(self, channel_n=16, hidden_n=128):
        super(NCAModel, self).__init__()
        self.channel_n = channel_n
        
        # Perception network: 3x3 Sobel filters for gradient detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Reshape for convolution
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
        
        # The perception conv layer has fixed weights (Sobel filters)
        self.perceive = nn.Conv2d(self.channel_n, self.channel_n * 2, 3, padding=1, bias=False, groups=self.channel_n)
        self.perceive.weight = nn.Parameter(torch.cat([sobel_x, sobel_y], dim=0), requires_grad=False)
        
        # Update network: A simple 2-layer MLP
        self.update_net = nn.Sequential(
            nn.Linear(self.channel_n * 3, hidden_n),
            nn.ReLU(),
            nn.Linear(hidden_n, self.channel_n, bias=False)
        )
        # Initialize the last layer's weights to zero
        self.update_net[-1].weight.data.fill_(0.0)

    def get_stochastic_update_mask(self, x):
        # Only update cells where the alpha channel > 0.1
        return (torch.rand_like(x[:, 3:4, :, :]) < 0.5) & (x[:, 3:4, :, :] > 0.1)

    def get_living_mask(self, x):
        # Cells are "alive" if their alpha channel > 0.1
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def forward(self, x, steps=64, return_sequence=False):
        if return_sequence:
            sequence = [x.clone()]

        for _ in range(steps):
            pre_life_mask = self.get_living_mask(x)
            
            perception_vectors = self.perceive(x)
            
            # Combine original state with perception vectors
            # Input to the update network is original cell state + perceived state
            combined_input = torch.cat([x, perception_vectors], dim=1).permute(0, 2, 3, 1)
            
            # Get the update delta from the network
            delta = self.update_net(combined_input).permute(0, 3, 1, 2)
            
            # Apply stochastic updates
            update_mask = self.get_stochastic_update_mask(x)
            x = x + delta * update_mask.float()
            
            # Ensure cells "die" if they are not surrounded by living cells
            post_life_mask = self.get_living_mask(x)
            life_mask = pre_life_mask & post_life_mask
            x = x * life_mask.float()
            
            if return_sequence:
                sequence.append(x.clone())

        return (x, sequence) if return_sequence else x

    def to_rgba(self, x):
        # The first 4 channels are interpreted as RGBA
        return x[:, :4, :, :]
    
    def get_seed(self, batch_size=1, size=128, device='cpu'):
        # Create a seed in the center of the grid
        seed = torch.zeros(batch_size, self.channel_n, size, size, device=device)
        seed[:, 3:, size // 2, size // 2] = 1.0 # Set alpha and hidden channels to 1
        return seed

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
    print(f"Running NCA test on device: {DEVICE}")
    # --- End Device Configuration ---

    # A simple test to verify the NCA model
    nca = NCAModel().to(DEVICE)
    seed = nca.get_seed(batch_size=1, size=64, device=DEVICE)
    
    # Grow the seed for a few steps
    output_grid, sequence = nca(seed, steps=32, return_sequence=True)
    
    final_image = nca.to_rgba(output_grid)
    
    print(f"Seed shape: {seed.shape}")
    print(f"Output grid shape: {output_grid.shape}")
    print(f"Final image RGBA shape: {final_image.shape}")
    
    # Visualize the growth sequence
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, (ax, frame) in enumerate(zip(axes.flat, sequence)):
        rgba_frame = nca.to_rgba(frame).detach().cpu().numpy()[0]
        # Clamp and transpose for visualization
        img_to_show = np.clip(rgba_frame.transpose(1, 2, 0), 0, 1)
        ax.imshow(img_to_show)
        ax.set_title(f"Step {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.show() 