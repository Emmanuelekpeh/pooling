import torch
import matplotlib.pyplot as plt
from train_integrated import IntegratedNCA, W_DIM, DEVICE, IMG_SIZE

def debug_seeding_issue():
    """Debug why the NCA is achieving 100% coverage immediately"""
    print("🔍 DEBUGGING: Why is NCA achieving 100% coverage immediately?")
    print("=" * 60)
    
    nca = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=False).to(DEVICE)
    
    # Test different seed types
    seed_types = ["center", "distributed", "corners", "line"]
    
    for seed_type in seed_types:
        print(f"\n{seed_type.upper()} SEEDING:")
        print("-" * 30)
        
        seed = nca.get_seed(batch_size=1, size=IMG_SIZE, device=DEVICE, seed_type=seed_type)
        
        # Count initial alive cells
        initial_alive = (seed[0, 3, :, :] > 0.1).sum().item()
        initial_coverage = (initial_alive / (IMG_SIZE * IMG_SIZE)) * 100
        
        print(f"Initial alive cells: {initial_alive} / {IMG_SIZE*IMG_SIZE}")
        print(f"Initial coverage: {initial_coverage:.1f}%")
        
        # Visualize the alpha channel (alive cells)
        alpha_channel = seed[0, 3, :, :].detach().cpu().numpy()
        
        # Show where cells are alive
        alive_positions = torch.where(seed[0, 3, :, :] > 0.1)
        if len(alive_positions[0]) > 0:
            print(f"Alive positions (first 10): {list(zip(alive_positions[0][:10].tolist(), alive_positions[1][:10].tolist()))}")
        
        # Check if any distributed seeds are overlapping/covering too much
        if seed_type == "distributed":
            print(f"Distributed seed details:")
            nca_temp = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=False)
            positions = nca_temp._get_distributed_positions(IMG_SIZE, 5)
            print(f"  Seed positions: {positions}")
            
            # Check radius coverage
            for i, (x, y) in enumerate(positions):
                radius = 2  # typical radius
                cells_in_radius = (2 * radius + 1) ** 2
                print(f"  Seed {i+1} at ({x},{y}): covers ~{cells_in_radius} cells with radius {radius}")
        
        # Save visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Alpha channel (alive cells)
        ax1.imshow(alpha_channel, cmap='Reds', vmin=0, vmax=1)
        ax1.set_title(f'{seed_type.title()} Seeding - Alpha Channel')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        
        # RGB channels (visual appearance)
        rgb = seed[0, :3, :, :].detach().cpu()
        rgb = torch.tanh(rgb)  # Apply activation
        rgb = (rgb + 1) / 2    # Normalize to [0,1]
        ax2.imshow(rgb.permute(1, 2, 0))
        ax2.set_title(f'{seed_type.title()} Seeding - RGB')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        
        plt.tight_layout()
        plt.savefig(f'debug_seed_{seed_type}.png', dpi=150, bbox_inches='tight')
        print(f"  💾 Saved visualization: debug_seed_{seed_type}.png")
    
    # Test what happens during a single NCA step
    print(f"\n" + "=" * 60)
    print("DEBUGGING: Single NCA Step")
    print("=" * 60)
    
    # Use center seed for simplicity
    seed = nca.get_seed(batch_size=1, size=IMG_SIZE, device=DEVICE, seed_type="center")
    initial_alive = (seed[0, 3, :, :] > 0.1).sum().item()
    
    # Create a simple w vector
    w = torch.randn(1, W_DIM).to(DEVICE)
    
    print(f"Before NCA step: {initial_alive} alive cells")
    
    # Run ONE step manually to see what happens
    with torch.no_grad():
        x = seed.clone()
        
        alive_mask = (x[:, 3:4, :, :] > 0.1).float()
        print(f"Alive mask sum: {alive_mask.sum().item()}")
        
        # Perception
        perceived = nca.perceive(x)
        perceived = perceived.permute(0, 2, 3, 1).reshape(-1, nca.channel_n * 3)
        print(f"Perceived shape: {perceived.shape}")
        
        # Conditioning
        conditioning_expanded = w.unsqueeze(1).unsqueeze(1).repeat(1, x.shape[2], x.shape[3], 1)
        conditioning_reshaped = conditioning_expanded.reshape(-1, w.shape[1])
        print(f"Conditioning shape: {conditioning_reshaped.shape}")
        
        # Update
        update_input = torch.cat([perceived, conditioning_reshaped], dim=1)
        print(f"Update input shape: {update_input.shape}")
        
        ds = nca.update_net(update_input)
        ds = ds.reshape(x.shape[0], x.shape[2], x.shape[3], nca.channel_n).permute(0, 3, 1, 2)
        print(f"Delta shape: {ds.shape}")
        print(f"Delta stats: min={ds.min().item():.4f}, max={ds.max().item():.4f}, mean={ds.mean().item():.4f}")
        
        # Apply update
        update_mask = alive_mask
        x = x + ds * update_mask * 0.1
        
        # Life dynamics
        neighbor_life = torch.nn.functional.max_pool2d(alive_mask, kernel_size=3, stride=1, padding=1)
        life_mask = (neighbor_life > 0.001).float()
        x = x * life_mask
        x[:, 3:4, :, :] = torch.clamp(x[:, 3:4, :, :] + 0.01 * alive_mask, 0, 1.2)
        
        after_alive = (x[0, 3, :, :] > 0.1).sum().item()
        print(f"After 1 NCA step: {after_alive} alive cells")
        print(f"Growth: {after_alive - initial_alive} cells")
        
        # Check if alpha values are exploding
        alpha_stats = x[0, 3, :, :]
        print(f"Alpha channel stats: min={alpha_stats.min().item():.4f}, max={alpha_stats.max().item():.4f}, mean={alpha_stats.mean().item():.4f}")
        
        # Count cells above different thresholds
        above_01 = (alpha_stats > 0.1).sum().item()
        above_05 = (alpha_stats > 0.5).sum().item() 
        above_10 = (alpha_stats > 1.0).sum().item()
        print(f"Cells above threshold: >0.1: {above_01}, >0.5: {above_05}, >1.0: {above_10}")

if __name__ == "__main__":
    debug_seeding_issue() 