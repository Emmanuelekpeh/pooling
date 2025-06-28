import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

# Import our simple growth NCA (we'll create a simplified version here)
# from simple_growth_nca import SimpleGrowthNCA

class SimpleGrowthNCA(nn.Module):
    """Clean, simple NCA focused purely on growth - adapted for single image training"""
    def __init__(self, n_channels=8):
        super().__init__()
        self.n_channels = n_channels
        
        # Simple perception - using Sobel filters like the breakthrough version
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Simple update network - exactly like breakthrough version
        self.update_net = nn.Sequential(
            nn.Conv2d(n_channels * 3, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, n_channels, 1)
        )
        
        # Growth parameters - key breakthrough parameters
        self.growth_rate = nn.Parameter(torch.tensor(0.3))
        self.alive_threshold = 0.1
        
        # Initialize for growth - exactly like breakthrough
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.05)  # Slight positive bias
    
    def perceive(self, x):
        """Simple perception - exactly like breakthrough version"""
        sobel_x_expanded = self.sobel_x.repeat(self.n_channels, 1, 1, 1)
        sobel_y_expanded = self.sobel_y.repeat(self.n_channels, 1, 1, 1)
        
        dx = F.conv2d(x, sobel_x_expanded, groups=self.n_channels, padding=1)
        dy = F.conv2d(x, sobel_y_expanded, groups=self.n_channels, padding=1)
        
        return torch.cat([x, dx, dy], dim=1)
    
    def forward(self, x, growth_boost=1.0):
        """Clean forward pass without in-place operations - breakthrough version"""
        # Perception
        perceived = self.perceive(x)
        
        # Update
        dx = self.update_net(perceived)
        
        # Apply growth rate with boost
        dx = dx * self.growth_rate * growth_boost
        
        # Growth encouragement for low populations - key breakthrough mechanism
        alive_mask = (x[:, 3:4] > self.alive_threshold).float()
        alive_ratio = alive_mask.mean()
        
        if alive_ratio < 0.2:
            # Emergency growth boost - exactly like breakthrough
            neighbor_alive = F.avg_pool2d(alive_mask, 3, stride=1, padding=1)
            growth_zones = (neighbor_alive > 0.01).float()
            
            # Create growth boost tensor
            growth_boost_tensor = torch.zeros_like(dx)
            growth_boost_tensor[:, 3:4] = growth_zones * 0.2  # Alpha boost
            
            dx = dx + growth_boost_tensor
        
        # Update state
        new_x = x + dx
        
        # Simple bounds - avoid in-place operations
        new_x = torch.clamp(new_x, -1, 1)
        alpha_clamped = torch.clamp(new_x[:, 3:4], 0, 1)  # Alpha in [0,1]
        new_x = torch.cat([new_x[:, :3], alpha_clamped, new_x[:, 4:]], dim=1)
        
        return new_x

def load_single_image(image_path, size=64):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] range
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def create_seed(size, device, seed_type="center"):
    """Create initial seed for NCA - using breakthrough parameters"""
    seed = torch.zeros(1, 8, size, size, device=device)  # 8 channels like breakthrough
    
    if seed_type == "center":
        center = size // 2
        radius = 3
        # Create circular seed
        y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
        dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
        mask = dist <= radius
        
        # Set RGB channels with some color - stronger initial values like breakthrough
        seed[0, 0, mask] = 0.5  # Red
        seed[0, 1, mask] = 0.3  # Green  
        seed[0, 2, mask] = 0.7  # Blue
        seed[0, 3, mask] = 1.0  # Alpha (alive) - strong like breakthrough
        
        # Initialize hidden channels
        for i in range(4, 8):
            seed[0, i, mask] = torch.randn(mask.sum(), device=device) * 0.1
    
    return seed

def get_progressive_growth_boost(epoch, total_epochs=150):
    """Progressive growth curriculum - the key breakthrough insight"""
    # Adapted for single image training (150 epochs)
    if epoch < total_epochs * 0.2:  # First 20% (30 epochs)
        return 2.0  # Strong early growth
    elif epoch < total_epochs * 0.4:  # Next 20% (60 epochs)
        return 1.5  # Moderate growth
    elif epoch < total_epochs * 0.6:  # Next 20% (90 epochs)
        return 1.2  # Light boost
    else:  # Final 40% (120-150 epochs)
        return 1.0  # Normal growth

def visualize_results(target, outputs, losses, alive_ratios, save_path="single_image_test_results.png"):
    """Create visualization of training results"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Convert target to displayable format
    target_img = (target[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
    target_img = np.clip(target_img, 0, 1)
    
    # Show target
    axes[0, 0].imshow(target_img)
    axes[0, 0].set_title("Target Image")
    axes[0, 0].axis('off')
    
    # Show NCA outputs at different stages
    stages = [0, len(outputs)//4, len(outputs)//2, 3*len(outputs)//4, -1]
    stage_names = ["Initial", "25%", "50%", "75%", "Final"]
    
    for i, (stage, name) in enumerate(zip(stages, stage_names)):
        if i == 0:  # Skip initial for first column (target is there)
            continue
        output = outputs[stage]
        # Convert to RGB and displayable format
        rgb = output[0, :3].cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb + 1) / 2
        rgb = np.clip(rgb, 0, 1)
        
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"NCA Output ({name})")
        axes[0, i].axis('off')
    
    # Plot training curves
    axes[1, 0].plot(losses)
    axes[1, 0].set_title("Training Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE Loss")
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(alive_ratios)
    axes[1, 1].set_title("Alive Cell Ratio")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].grid(True)
    
    # Show alpha channel progression
    alpha_final = outputs[-1][0, 3].cpu().numpy()
    axes[1, 2].imshow(alpha_final, cmap='viridis')
    axes[1, 2].set_title("Final Alpha Channel")
    axes[1, 2].axis('off')
    
    # Show growth pattern
    if len(alive_ratios) > 10:
        growth_rate = np.diff(alive_ratios)
        axes[1, 3].plot(growth_rate)
        axes[1, 3].set_title("Growth Rate")
        axes[1, 3].set_xlabel("Epoch")
        axes[1, 3].set_ylabel("Change in Alive Ratio")
        axes[1, 3].grid(True)
    
    # Show loss vs alive ratio correlation
    if len(losses) == len(alive_ratios):
        axes[1, 4].scatter(alive_ratios, losses, alpha=0.6, s=10)
        axes[1, 4].set_title("Loss vs Alive Ratio")
        axes[1, 4].set_xlabel("Alive Ratio")
        axes[1, 4].set_ylabel("Loss")
        axes[1, 4].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to {save_path}")
    return fig

def test_single_image_nca(image_path, epochs=150, steps_per_epoch=15):
    """Test NCA training on a single image - using breakthrough parameters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Testing with image: {os.path.basename(image_path)}")
    
    # Load target image - start with smaller size like breakthrough
    target = load_single_image(image_path, size=64).to(device)
    print(f"Target image shape: {target.shape}")
    
    # Initialize NCA model - using breakthrough parameters
    model = SimpleGrowthNCA(n_channels=8).to(device)  # 8 channels like breakthrough
    optimizer = optim.Adam(model.parameters(), lr=0.002)  # Same LR as breakthrough
    
    # Training tracking
    losses = []
    alive_ratios = []
    outputs = []
    growth_boosts = []
    
    print(f"Training for {epochs} epochs with {steps_per_epoch} steps per epoch...")
    print("🌱 Using Progressive Growth Curriculum (breakthrough approach)")
    
    for epoch in range(epochs):
        model.train()
        
        # Get progressive growth boost - THE KEY BREAKTHROUGH
        growth_boost = get_progressive_growth_boost(epoch, epochs)
        growth_boosts.append(growth_boost)
        
        # Create fresh seed each epoch
        seed = create_seed(64, device, "center")
        
        # Run NCA for specified steps - using breakthrough step count
        x = seed
        for step in range(steps_per_epoch):
            x = model(x, growth_boost=growth_boost)  # Apply growth boost
        
        # Calculate loss (only on RGB channels)
        target_rgb = target[:, :3]  # Only RGB channels
        output_rgb = x[:, :3]       # Only RGB channels
        loss = F.mse_loss(output_rgb, target_rgb)
        
        # Calculate alive ratio
        alive_mask = (x[:, 3] > 0.1).float()
        alive_ratio = alive_mask.mean().item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping - same as breakthrough
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Track metrics
        losses.append(loss.item())
        alive_ratios.append(alive_ratio)
        
        # Save outputs periodically
        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            with torch.no_grad():
                outputs.append(x.clone())
        
        # Print progress with growth boost info
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f}, Alive = {alive_ratio:.4f}, Boost = {growth_boost:.1f}")
        
        # Early success check - like breakthrough
        if alive_ratio > 0.15 and epoch > 20:
            print(f"   🎉 Growth success! Alive ratio: {alive_ratio:.3f} at epoch {epoch}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Test with longer run
        seed = create_seed(64, device, "center")
        x = seed
        final_outputs = [x.clone()]
        
        for step in range(steps_per_epoch * 2):  # Run longer
            x = model(x, growth_boost=1.0)  # Normal growth for final test
            if step % 5 == 0:
                final_outputs.append(x.clone())
        
        final_loss = F.mse_loss(x[:, :3], target[:, :3])
        final_alive_ratio = (x[:, 3] > 0.1).float().mean().item()
    
    # Create visualization
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    results_path = f"single_image_test_{image_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig = visualize_results(target, outputs, losses, alive_ratios, results_path)
    
    # Calculate success metrics like breakthrough
    max_alive = max(alive_ratios)
    final_training_alive = alive_ratios[-1]
    growth_success = max_alive > 0.12
    sustained_growth = final_alive_ratio > 0.08
    
    # Save detailed results
    results = {
        'image_path': image_path,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'final_loss': float(final_loss.item()),
        'final_alive_ratio': float(final_alive_ratio),
        'min_loss': float(min(losses)),
        'max_alive_ratio': float(max_alive),
        'final_training_alive': float(final_training_alive),
        'losses': [float(x) for x in losses],
        'alive_ratios': [float(x) for x in alive_ratios],
        'growth_boosts': [float(x) for x in growth_boosts],
        'training_summary': {
            'converged': bool(final_loss < 0.1),
            'stable_growth': bool(abs(alive_ratios[-1] - alive_ratios[-10]) < 0.05 if len(alive_ratios) >= 10 else False),
            'growth_achieved': bool(growth_success),
            'sustained_growth': bool(sustained_growth)
        },
        'breakthrough_metrics': {
            'max_alive': float(max_alive),
            'final_training_alive': float(final_training_alive),
            'final_test_alive': float(final_alive_ratio),
            'growth_success': bool(growth_success),
            'sustained_growth': bool(sustained_growth)
        }
    }
    
    results_json_path = f"single_image_test_{image_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== BREAKTHROUGH ANALYSIS ===")
    print(f"Max alive ratio: {max_alive:.4f}")
    print(f"Final training alive: {final_training_alive:.4f}")
    print(f"Final test alive: {final_alive_ratio:.4f}")
    print(f"Final loss: {final_loss.item():.6f}")
    print(f"Growth success: {'✅' if growth_success else '❌'} (target: >0.12)")
    print(f"Sustained growth: {'✅' if sustained_growth else '❌'} (target: >0.08)")
    
    if growth_success and sustained_growth:
        print(f"\n🎉 SUCCESS! Breakthrough approach achieved stable growth for art generation!")
    elif growth_success:
        print(f"\n🔄 PARTIAL: Achieved growth but sustainability needs improvement")
    else:
        print(f"\n❌ CHALLENGE: Still struggling - may need parameter adjustment")
    
    print(f"Results saved to: {results_path}")
    print(f"Detailed data saved to: {results_json_path}")
    
    return model, results, fig

if __name__ == "__main__":
    # Test with a traditional ukiyo-e image
    image_path = "./data/ukiyo-e-small/Along the Shore of Yènoshim.jpg"
    
    if os.path.exists(image_path):
        print("🧬 SimpleGrowthNCA Single Image Test")
        print("   Using breakthrough parameters and progressive growth curriculum")
        print("   Goal: Stable growth for Ukiyo-e art generation")
        print()
        
        model, results, fig = test_single_image_nca(image_path, epochs=150, steps_per_epoch=15)
        plt.show()
    else:
        print(f"Image not found: {image_path}")
        print("Available images:")
        for f in os.listdir("./data/ukiyo-e-small"):
            if f.endswith(('.jpg', '.JPG', '.png', '.PNG')):
                print(f"  {f}")