import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated import IntegratedNCA, IntegratedGenerator, Discriminator, CrossEvaluator
from train_integrated import Z_DIM, W_DIM, DEVICE, IMG_SIZE
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import time

from src.models.integrated import IntegratedNCA
from src.data_loader.loader import ImageDataset

def test_enhanced_penalties():
    """Test the new enhanced penalty system without matching loss"""
    print("Testing Enhanced Penalty System...")
    
    # Create models
    generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
    nca_model = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=True).to(DEVICE)
    discriminator = Discriminator(IMG_SIZE).to(DEVICE)
    gen_evaluator = CrossEvaluator(IMG_SIZE).to(DEVICE)
    nca_evaluator = CrossEvaluator(IMG_SIZE).to(DEVICE)
    
    # Create test inputs
    batch_size = 2
    noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
    real_imgs = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # Generate fake images
    fake_imgs_gen, w = generator(noise, return_w=True)
    seed = nca_model.get_seed(batch_size=batch_size, size=IMG_SIZE, device=DEVICE)
    nca_output_grid = nca_model(seed, w, steps=32, target_img=real_imgs)
    nca_rgba = nca_model.to_rgba(nca_output_grid)
    fake_imgs_nca = nca_rgba[:, :3, :, :] * 2.0 - 1.0
    
    # Get discriminator predictions
    disc_fake_gen_pred = discriminator(fake_imgs_gen)
    disc_fake_nca_pred = discriminator(fake_imgs_nca)
    
    # Get cross-evaluation scores
    gen_scores_nca = nca_evaluator(fake_imgs_gen)
    nca_scores_gen = gen_evaluator(fake_imgs_nca)
    
    print(f"Generator scores from NCA evaluator: {gen_scores_nca.mean().item():.4f}")
    print(f"NCA scores from Generator evaluator: {nca_scores_gen.mean().item():.4f}")
    
    # Test Enhanced Penalty System
    epoch = 25  # Mid-training epoch
    
    # 1. Exponential penalty scaling
    quality_threshold = 0.3
    gen_penalty_scaling = torch.exp(-gen_scores_nca * 5)
    nca_penalty_scaling = torch.exp(-nca_scores_gen * 5)
    
    print(f"\n--- Exponential Penalty Scaling ---")
    print(f"Generator penalty scaling: {gen_penalty_scaling.mean().item():.4f}")
    print(f"NCA penalty scaling: {nca_penalty_scaling.mean().item():.4f}")
    
    # 2. Threshold penalties
    gen_below_threshold = (gen_scores_nca < quality_threshold).float()
    nca_below_threshold = (nca_scores_gen < quality_threshold).float()
    threshold_penalty_gen = gen_below_threshold * 10.0
    threshold_penalty_nca = nca_below_threshold * 10.0
    
    print(f"\n--- Threshold Penalties ---")
    print(f"Generator below threshold: {gen_below_threshold.sum().item()}/{batch_size}")
    print(f"NCA below threshold: {nca_below_threshold.sum().item()}/{batch_size}")
    print(f"Generator threshold penalty: {threshold_penalty_gen.mean().item():.4f}")
    print(f"NCA threshold penalty: {threshold_penalty_nca.mean().item():.4f}")
    
    # 3. Progressive penalty increase
    epoch_multiplier = min((epoch + 1) / 50, 3.0)
    print(f"\n--- Progressive Penalty ---")
    print(f"Epoch multiplier (epoch {epoch}): {epoch_multiplier:.2f}")
    
    # 4. Feature matching penalty (replaces pixel matching)
    with torch.no_grad():
        disc_features_real = discriminator.model[:-1](real_imgs)
        disc_features_nca = discriminator.model[:-1](fake_imgs_nca)
        disc_features_gen = discriminator.model[:-1](fake_imgs_gen)
    
    feature_match_loss_nca = F.mse_loss(disc_features_nca.mean(dim=0), disc_features_real.mean(dim=0))
    feature_match_loss_gen = F.mse_loss(disc_features_gen.mean(dim=0), disc_features_real.mean(dim=0))
    
    print(f"\n--- Feature Matching (vs Pixel Matching) ---")
    print(f"Generator feature matching loss: {feature_match_loss_gen.item():.4f}")
    print(f"NCA feature matching loss: {feature_match_loss_nca.item():.4f}")
    
    # Compare with old pixel matching loss (for reference)
    pixel_match_loss = F.mse_loss(fake_imgs_nca, fake_imgs_gen.detach())
    print(f"Old pixel matching loss (removed): {pixel_match_loss.item():.4f}")
    
    # 5. Overconfidence penalty
    overconfidence_penalty_gen = torch.relu(gen_scores_nca - 0.9).mean() * 2.0
    overconfidence_penalty_nca = torch.relu(nca_scores_gen - 0.9).mean() * 2.0
    
    print(f"\n--- Overconfidence Penalties ---")
    print(f"Generator overconfidence penalty: {overconfidence_penalty_gen.item():.4f}")
    print(f"NCA overconfidence penalty: {overconfidence_penalty_nca.item():.4f}")
    
    # Final combined penalties
    loss_gen_mutual = (gen_penalty_scaling.mean() + 
                      threshold_penalty_gen.mean() + 
                      feature_match_loss_gen * 2.0 +
                      overconfidence_penalty_gen) * epoch_multiplier
    
    loss_nca_mutual = (nca_penalty_scaling.mean() + 
                      threshold_penalty_nca.mean() + 
                      feature_match_loss_nca * 2.0 +
                      overconfidence_penalty_nca) * epoch_multiplier
    
    print(f"\n--- FINAL PENALTIES ---")
    print(f"Generator total penalty: {loss_gen_mutual.item():.4f}")
    print(f"NCA total penalty: {loss_nca_mutual.item():.4f}")
    print(f"Penalty ratio (NCA/Gen): {(loss_nca_mutual.item() / max(loss_gen_mutual.item(), 1e-8)):.2f}")
    
    # Test different quality scenarios
    print(f"\n--- Testing Different Quality Scenarios ---")
    
    # Low quality scenario
    low_scores = torch.tensor([0.1, 0.2]).to(DEVICE)
    low_penalty = torch.exp(-low_scores * 5).mean()
    low_threshold = (low_scores < 0.3).float().mean() * 10.0
    print(f"Low quality (0.1-0.2): Exp penalty={low_penalty:.4f}, Threshold penalty={low_threshold:.4f}")
    
    # Medium quality scenario  
    med_scores = torch.tensor([0.4, 0.5]).to(DEVICE)
    med_penalty = torch.exp(-med_scores * 5).mean()
    med_threshold = (med_scores < 0.3).float().mean() * 10.0
    print(f"Medium quality (0.4-0.5): Exp penalty={med_penalty:.4f}, Threshold penalty={med_threshold:.4f}")
    
    # High quality scenario
    high_scores = torch.tensor([0.8, 0.9]).to(DEVICE)
    high_penalty = torch.exp(-high_scores * 5).mean()
    high_threshold = (high_scores < 0.3).float().mean() * 10.0
    high_overconf = torch.relu(high_scores - 0.9).mean() * 2.0
    print(f"High quality (0.8-0.9): Exp penalty={high_penalty:.4f}, Threshold penalty={high_threshold:.4f}, Overconf penalty={high_overconf:.4f}")
    
    print(f"\n✅ Enhanced penalty system test completed!")
    print(f"📈 Penalties scale exponentially with poor performance")
    print(f"🎯 Threshold penalties kick in below quality 0.3")
    print(f"📊 Feature matching replaces pixel matching for stability")
    print(f"⚖️ Progressive penalties increase over training epochs")
    print(f"🚫 No more matching loss constraint on NCA")

def test_proper_spatial_runoff():
    """Proper test of spatial runoff starting from realistic seeds"""
    print("🎯 PROPER Test: Enhanced Runoff for Spatial Exploration")
    print("=" * 60)
    
    # Create NCA model
    nca = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=False).to(DEVICE)
    nca.eval()
    
    # Simple conditioning
    w = torch.randn(1, W_DIM).to(DEVICE)
    
    # Test different seed configurations
    seed_types = ["center", "distributed", "corners", "line"]
    
    results = {}
    
    for seed_type in seed_types:
        print(f"\n{seed_type.upper()} SEEDING:")
        print("-" * 30)
        
        # Start with proper seed
        seed = nca.get_seed(batch_size=1, size=IMG_SIZE, device=DEVICE, seed_type=seed_type)
        
        # Verify actual initial coverage
        initial_rgba = nca.to_rgba(seed)
        initial_alive = (initial_rgba[:, 3:4, :, :] > 0.1).float()
        initial_count = initial_alive.sum().item()
        initial_coverage = initial_count / 4096 * 100
        
        print(f"  Initial: {initial_count:.0f} cells ({initial_coverage:.1f}% coverage)")
        
        # Track growth over steps
        step_data = []
        current_x = seed.clone()
        
        test_steps = [0, 10, 20, 30, 40, 50]
        
        for i, target_steps in enumerate(test_steps):
            if i == 0:
                # Initial state
                rgba = nca.to_rgba(current_x)
            else:
                # Grow incrementally
                steps_to_grow = target_steps - test_steps[i-1]
                current_x = nca(current_x, w, steps=steps_to_grow)
                rgba = nca.to_rgba(current_x)
            
            # Measure alive cells and spatial characteristics
            alive_mask = (rgba[:, 3:4, :, :] > 0.1).float()
            alive_count = alive_mask.sum().item()
            coverage = alive_count / 4096 * 100
            
            # Measure spatial spread (standard deviation of positions)
            if alive_count > 1:
                alive_positions = torch.nonzero(alive_mask[0, 0, :, :], as_tuple=False).float()
                center_x = alive_positions[:, 0].mean()
                center_y = alive_positions[:, 1].mean()
                spread_x = ((alive_positions[:, 0] - center_x) ** 2).mean().sqrt()
                spread_y = ((alive_positions[:, 1] - center_y) ** 2).mean().sqrt()
                spatial_spread = (spread_x + spread_y) / 2
                
                # Measure how "clustered" vs "dispersed" the pattern is
                # Using nearest neighbor distances
                if alive_count > 5:
                    # Sample some positions for efficiency
                    sample_size = min(100, len(alive_positions))
                    sample_idx = torch.randperm(len(alive_positions))[:sample_size]
                    sample_pos = alive_positions[sample_idx]
                    
                    # Calculate average distance to nearest neighbors
                    distances = torch.cdist(sample_pos, sample_pos)
                    distances[distances == 0] = float('inf')  # Remove self-distances
                    nearest_dist = distances.min(dim=1)[0].mean()
                else:
                    nearest_dist = 0.0
            else:
                spatial_spread = 0.0
                nearest_dist = 0.0
            
            step_data.append({
                'step': target_steps,
                'alive_count': alive_count,
                'coverage': coverage,
                'spatial_spread': spatial_spread,
                'nearest_neighbor_dist': nearest_dist.item() if isinstance(nearest_dist, torch.Tensor) else nearest_dist
            })
            
            print(f"  Step {target_steps:2d}: {alive_count:4.0f} cells ({coverage:5.1f}%), "
                  f"Spread: {spatial_spread:5.2f}, NN-dist: {nearest_dist:.2f}")
        
        results[seed_type] = step_data
    
    # Analyze results
    print(f"\n{'=' * 60}")
    print("📊 RUNOFF ANALYSIS:")
    print(f"{'=' * 60}")
    
    for seed_type, data in results.items():
        initial = data[0]
        final = data[-1]
        
        growth_rate = (final['alive_count'] - initial['alive_count']) / len(data)
        spread_increase = final['spatial_spread'] - initial['spatial_spread']
        
        print(f"\n{seed_type.upper()}:")
        print(f"  Growth rate: {growth_rate:6.1f} cells/step")
        print(f"  Final coverage: {final['coverage']:5.1f}%")
        print(f"  Spread increase: {spread_increase:5.2f}")
        print(f"  Final NN-distance: {final['nearest_neighbor_dist']:5.2f}")
        
        # Check for healthy spatial exploration
        if final['coverage'] > 25 and spread_increase > 5:
            print(f"  ✅ Good spatial exploration!")
        elif final['coverage'] > 10:
            print(f"  ⚠️  Moderate growth, limited spread")
        else:
            print(f"  ❌ Poor growth/spread")
    
    return results

def visualize_runoff_comparison(results):
    """Create visualizations comparing different seed types"""
    print(f"\n📈 Creating runoff comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics = ['alive_count', 'coverage', 'spatial_spread', 'nearest_neighbor_dist']
    titles = ['Alive Cell Count', 'Coverage %', 'Spatial Spread', 'Nearest Neighbor Distance']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for seed_type, data in results.items():
            steps = [d['step'] for d in data]
            values = [d[metric] for d in data]
            axes[i].plot(steps, values, marker='o', label=seed_type, linewidth=2)
        
        axes[i].set_xlabel('Steps')
        axes[i].set_ylabel(title)
        axes[i].set_title(f'{title} Over Time')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runoff_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Runoff comparison saved as 'runoff_comparison.png'")

class EnhancedNCA(nn.Module):
    def __init__(self, channel_n=16, hidden_n=128, w_dim=128, activation='gelu'):
        super().__init__()
        self.channel_n = channel_n
        self.hidden_n = hidden_n
        self.w_dim = w_dim
        self.activation = activation
        
        # Perception network (same as original)
        self.perception_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_n, hidden_n, 1),
            nn.ReLU()
        )
        
        # Update network with selected activation
        self._setup_update_net()
        
    def _setup_update_net(self):
        if self.activation == 'gelu':
            self.update_net = nn.Sequential(
                nn.Conv2d(self.hidden_n + self.w_dim, self.hidden_n, 1),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_n),
                nn.Conv2d(self.hidden_n, self.channel_n, 1),
                nn.Tanh()
            )
        elif self.activation == 'leaky_relu':
            self.update_net = nn.Sequential(
                nn.Conv2d(self.hidden_n + self.w_dim, self.hidden_n, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(self.hidden_n),
                nn.Conv2d(self.hidden_n, self.channel_n, 1),
                nn.Tanh()
            )
    
    def get_seed(self, batch_size, size, device):
        # Create a seed pattern (center seed)
        x = torch.zeros(batch_size, self.channel_n, size, size, device=device)
        x[:, 3:, size//2-1:size//2+1, size//2-1:size//2+1] = 1.0
        return x
    
    def perceive(self, x):
        # Stack channels: normal, shifted up/down, shifted left/right
        y = torch.cat((x,
            torch.roll(x, shifts=1, dims=2),
            torch.roll(x, shifts=-1, dims=2),
            torch.roll(x, shifts=1, dims=3),
            torch.roll(x, shifts=-1, dims=3)), 1)
        return self.perception_net(y)
    
    def forward(self, x, w, steps):
        batch_size = x.shape[0]
        w = w.view(batch_size, self.w_dim, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        for _ in range(steps):
            y = self.perceive(x)
            # Concatenate w along channel dimension
            y = torch.cat([y, w], dim=1)
            dx = self.update_net(y)
            x = x + dx
            
        return x

def train_step(model, optimizer, x, w, target, steps=50):
    optimizer.zero_grad()
    output = model(x, w, steps)
    loss = torch.mean((output - target) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item()

def run_comparison(batch_size=4, img_size=64, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models with different activations
    models = {
        'gelu': EnhancedNCA(activation='gelu').to(device),
        'leaky_relu': EnhancedNCA(activation='leaky_relu').to(device)
    }
    
    optimizers = {
        name: optim.Adam(model.parameters(), lr=1e-4)
        for name, model in models.items()
    }
    
    # Training history
    history = {name: [] for name in models.keys()}
    
    # Generate random target images
    target = torch.randn(batch_size, 16, img_size, img_size).to(device)
    target = torch.tanh(target)  # Normalize targets
    
    print("Starting training comparison...")
    for epoch in range(epochs):
        for name, model in models.items():
            # Get seed
            x = model.get_seed(batch_size, img_size, device)
            # Generate latent code
            w = torch.randn(batch_size, 128).to(device)
            
            # Training step
            loss = train_step(model, optimizers[name], x, w, target)
            history[name].append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, {name.upper()} Loss: {loss:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, losses in history.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Activation Function Comparison')
    plt.legend()
    plt.savefig('activation_comparison.png')
    plt.close()

if __name__ == "__main__":
    print("🚀 Testing Enhanced Runoff Mechanisms")
    print("=" * 60)
    
    results = test_proper_spatial_runoff()
    visualize_runoff_comparison(results)
    
    print(f"\n🎉 Enhanced runoff testing completed!")
    print("Look for improved spatial exploration in the results above.")
    
    run_comparison() 