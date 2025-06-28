#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

# Advanced Homeostatic NCA with Memory and Attention
class HomeostaticNCA(nn.Module):
    def __init__(self, n_channels=16, n_memory_channels=8, n_hardware_channels=4, n_attention_heads=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_memory_channels = n_memory_channels  # Private memory not visible to neighbors
        self.n_hardware_channels = n_hardware_channels  # Immutable regulatory parameters
        self.n_attention_heads = n_attention_heads
        
        # Total channels: visible + memory + hardware
        total_channels = n_channels + n_memory_channels + n_hardware_channels
        
        # Perception kernels (Sobel filters for gradients)
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('identity', torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('laplacian', torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Perception network - processes local neighborhood
        perception_size = total_channels * 4  # 4 kernels per channel
        self.perception_net = nn.Sequential(
            nn.Linear(perception_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Attention mechanism for mode selection
        self.attention_embed = nn.Linear(n_hardware_channels, n_attention_heads)
        self.attention_temp = nn.Parameter(torch.tensor(1.0))
        
        # Multiple update pathways (attention heads)
        self.update_pathways = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_channels + n_memory_channels)  # Only update visible + memory, not hardware
            ) for _ in range(n_attention_heads)
        ])
        
        # Homeostatic controller
        self.homeostatic_controller = nn.Sequential(
            nn.Linear(1 + 1, 16),  # memory_mean + alive_ratio
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),  # Output scaling factor
            nn.Sigmoid()
        )
        
        # Initialize with "do nothing" behavior
        for pathway in self.update_pathways:
            nn.init.zeros_(pathway[-1].weight)
            nn.init.zeros_(pathway[-1].bias)

    def perceive(self, x):
        """Apply perception kernels to all channels"""
        batch_size, channels, height, width = x.shape
        
        # Apply kernels to each channel
        perceived = []
        for kernel in [self.identity, self.sobel_x, self.sobel_y, self.laplacian]:
            kernel_expanded = kernel.expand(channels, -1, -1, -1)
            conv_result = F.conv2d(x, kernel_expanded, groups=channels, padding=1)
            perceived.append(conv_result)
        
        # Concatenate all perceptions
        perception = torch.cat(perceived, dim=1)  # [batch, channels*4, height, width]
        return perception

    def forward(self, x, step_prob=0.5):
        batch_size, total_channels, height, width = x.shape
        
        # Split channels
        visible = x[:, :self.n_channels]
        memory = x[:, self.n_channels:self.n_channels + self.n_memory_channels]
        hardware = x[:, -self.n_hardware_channels:]
        
        # Calculate alive cells (alpha > 0.1)
        alive_mask = (visible[:, 3:4] > 0.1).float()
        alive_ratio = alive_mask.mean(dim=[2, 3], keepdim=True)
        
        # Perception
        perception = self.perceive(x)  # [batch, total_channels*4, height, width]
        
        # Reshape for processing
        perception_flat = perception.view(batch_size, -1, height * width).permute(0, 2, 1)  # [batch, hw, features]
        memory_flat = memory.view(batch_size, -1, height * width).permute(0, 2, 1)  # [batch, hw, memory]
        hardware_flat = hardware.view(batch_size, -1, height * width).permute(0, 2, 1)  # [batch, hw, hardware]
        
        # Process perception
        processed_perception = self.perception_net(perception_flat)  # [batch, hw, 64]
        
        # Attention mechanism - hardware determines which pathways to activate
        attention_logits = self.attention_embed(hardware_flat) / self.attention_temp  # [batch, hw, n_heads]
        attention_weights = F.softmax(attention_logits, dim=-1)  # [batch, hw, n_heads]
        
        # Apply pathways with attention weighting
        pathway_outputs = []
        for i, pathway in enumerate(self.update_pathways):
            pathway_out = pathway(processed_perception)  # [batch, hw, channels+memory]
            pathway_outputs.append(pathway_out)
        
        pathway_stack = torch.stack(pathway_outputs, dim=-1)  # [batch, hw, channels+memory, n_heads]
        
        # Weighted combination of pathways
        attention_expanded = attention_weights.unsqueeze(-2)  # [batch, hw, 1, n_heads]
        weighted_update = (pathway_stack * attention_expanded).sum(dim=-1)  # [batch, hw, channels+memory]
        
        # Homeostatic regulation
        alive_ratio_expanded = alive_ratio.expand(-1, -1, height, width).view(batch_size, height * width, 1)
        memory_mean = memory_flat.mean(dim=-1, keepdim=True)  # Average memory state
        homeostatic_input = torch.cat([memory_mean, alive_ratio_expanded], dim=-1)
        homeostatic_scale = self.homeostatic_controller(homeostatic_input)  # [batch, hw, 1]
        
        # Apply homeostatic scaling
        weighted_update = weighted_update * homeostatic_scale
        
        # Reshape back to spatial dimensions
        update_spatial = weighted_update.permute(0, 2, 1).view(batch_size, -1, height, width)
        visible_update = update_spatial[:, :self.n_channels]
        memory_update = update_spatial[:, self.n_channels:]
        
        # Stochastic update mask (asynchronous updates)
        if self.training:
            update_mask = (torch.rand_like(visible_update[:, :1]) < step_prob).float()
        else:
            update_mask = torch.ones_like(visible_update[:, :1])
        
        # Apply updates only to visible and memory channels (hardware is immutable)
        new_visible = visible + visible_update * update_mask
        new_memory = memory + memory_update * update_mask
        
        # Enforce alive mask - dead cells have zero state
        alive_extended = F.max_pool2d(alive_mask, 3, stride=1, padding=1)
        new_visible = new_visible * alive_extended
        new_memory = new_memory * alive_extended
        
        # Recombine all channels
        new_x = torch.cat([new_visible, new_memory, hardware], dim=1)
        
        return new_x

def create_homeostatic_hardware(size, device):
    """Create specialized hardware configurations for different regions"""
    hardware = torch.zeros(1, 4, size, size, device=device)
    
    # Input region hardware (bottom-left)
    input_region = torch.zeros(4)
    input_region[0] = 1.0  # Input marker
    input_region[1] = 0.8  # High growth promotion
    input_region[2] = 0.2  # Low death promotion
    input_region[3] = 0.5  # Moderate stability
    
    # Growth region hardware (center)
    growth_region = torch.zeros(4)
    growth_region[0] = 0.0  # Not input
    growth_region[1] = 0.6  # Moderate growth
    growth_region[2] = 0.4  # Moderate death
    growth_region[3] = 0.8  # High stability preference
    
    # Boundary region hardware (edges)
    boundary_region = torch.zeros(4)
    boundary_region[0] = 0.0  # Not input
    boundary_region[1] = 0.2  # Low growth
    boundary_region[2] = 0.8  # High death (boundary cleanup)
    boundary_region[3] = 0.3  # Low stability
    
    # Apply hardware configurations
    # Input region (bottom-left quadrant)
    hardware[0, :, size//2:, :size//2] = input_region.view(-1, 1, 1)
    
    # Boundary regions (edges)
    hardware[0, :, :5, :] = boundary_region.view(-1, 1, 1)  # Top
    hardware[0, :, -5:, :] = boundary_region.view(-1, 1, 1)  # Bottom
    hardware[0, :, :, :5] = boundary_region.view(-1, 1, 1)  # Left
    hardware[0, :, :, -5:] = boundary_region.view(-1, 1, 1)  # Right
    
    # Growth region (center, overriding some boundary)
    center_start = size // 4
    center_end = 3 * size // 4
    hardware[0, :, center_start:center_end, center_start:center_end] = growth_region.view(-1, 1, 1)
    
    return hardware

def initialize_homeostatic_state(size, device):
    """Initialize state with visible channels, memory channels, and hardware"""
    n_channels = 16
    n_memory = 8
    n_hardware = 4
    
    # Initialize visible channels (RGBA + hidden)
    visible = torch.zeros(1, n_channels, size, size, device=device)
    
    # Add seed in center
    center = size // 2
    visible[0, 3, center-1:center+2, center-1:center+2] = 1.0  # Alpha channel
    visible[0, 4:, center-1:center+2, center-1:center+2] = torch.randn(n_channels-4, 3, 3, device=device) * 0.1
    
    # Initialize memory channels (internal cell memory)
    memory = torch.zeros(1, n_memory, size, size, device=device)
    
    # Create hardware configuration
    hardware = create_homeostatic_hardware(size, device)
    
    # Combine all channels
    state = torch.cat([visible, memory, hardware], dim=1)
    
    return state

def test_homeostatic_nca():
    """Test the homeostatic NCA with different scenarios"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    size = 64
    n_channels = 16
    n_memory = 8
    n_hardware = 4
    
    # Initialize model
    model = HomeostaticNCA(n_channels=n_channels, n_memory_channels=n_memory, 
                          n_hardware_channels=n_hardware, n_attention_heads=4).to(device)
    
    print("Testing Advanced Homeostatic NCA System")
    print("=" * 50)
    print("Key Features:")
    print("- Attention-based update pathways")
    print("- Private memory channels")
    print("- Immutable hardware configuration")
    print("- Asynchronous stochastic updates")
    print("- Homeostatic population control")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        {"name": "Standard_Seed", "steps": 100},
        {"name": "High_Density_Start", "steps": 100},
        {"name": "Sparse_Start", "steps": 100},
        {"name": "Random_Noise_Start", "steps": 100},
        {"name": "Long_Term_Stability", "steps": 500}
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        
        model.eval()
        with torch.no_grad():
            # Initialize different starting conditions
            if scenario['name'] == "Standard_Seed":
                state = initialize_homeostatic_state(size, device)
            elif scenario['name'] == "High_Density_Start":
                state = initialize_homeostatic_state(size, device)
                # Add more alive cells
                center = size // 2
                state[0, 3, center-5:center+5, center-5:center+5] = 1.0
                state[0, 4:8, center-5:center+5, center-5:center+5] = torch.randn(4, 10, 10, device=device) * 0.2
            elif scenario['name'] == "Sparse_Start":
                state = initialize_homeostatic_state(size, device)
                # Reduce to single cell
                state[0, 3] = 0.0
                center = size // 2
                state[0, 3, center, center] = 1.0
                state[0, 4:8, center, center] = torch.randn(4, device=device) * 0.1
            elif scenario['name'] == "Random_Noise_Start":
                state = initialize_homeostatic_state(size, device)
                # Add random noise
                noise_mask = torch.rand(1, 1, size, size, device=device) < 0.1
                state[0, 3] = state[0, 3] + noise_mask.float() * 0.5
                state[0, 4:8] = state[0, 4:8] + torch.randn_like(state[0, 4:8]) * 0.1 * noise_mask
            else:  # Long_Term_Stability
                state = initialize_homeostatic_state(size, device)
            
            # Track evolution
            alive_counts = []
            states_to_save = []
            
            initial_alive = (state[:, 3:4] > 0.1).float().sum().item()
            alive_counts.append(initial_alive)
            
            if len(states_to_save) < 10:  # Save first few states
                states_to_save.append(state[:, :n_channels].cpu().numpy())
            
            # Run simulation
            for step in range(scenario['steps']):
                state = model(state, step_prob=1.0)  # Deterministic for testing
                
                # Track alive cells
                alive_count = (state[:, 3:4] > 0.1).float().sum().item()
                alive_counts.append(alive_count)
                
                # Save states periodically
                if step % (scenario['steps'] // 10) == 0 and len(states_to_save) < 10:
                    states_to_save.append(state[:, :n_channels].cpu().numpy())
                
                # Early termination if system dies or explodes
                if alive_count == 0:
                    print(f"  System died at step {step}")
                    break
                elif alive_count >= size * size * 0.8:  # 80% of grid
                    print(f"  System overgrew at step {step} (alive: {alive_count})")
                    break
            
            # Calculate statistics
            final_alive = alive_counts[-1]
            max_alive = max(alive_counts)
            min_alive = min(alive_counts[1:]) if len(alive_counts) > 1 else alive_counts[0]
            
            # Stability score (lower variance = more stable)
            if len(alive_counts) > 10:
                recent_counts = alive_counts[-10:]
                stability_score = 1.0 / (1.0 + np.var(recent_counts))
            else:
                stability_score = 0.0
            
            # Health classification
            if final_alive == 0:
                health = "DEAD"
            elif final_alive >= size * size * 0.8:
                health = "OVERGROWN"
            elif 50 <= final_alive <= 2000:
                health = "HEALTHY"
            else:
                health = "UNSTABLE"
            
            results[scenario['name']] = {
                'initial_alive': initial_alive,
                'final_alive': final_alive,
                'max_alive': max_alive,
                'min_alive': min_alive,
                'stability_score': stability_score,
                'health': health,
                'alive_counts': alive_counts,
                'steps_completed': len(alive_counts) - 1
            }
            
            print(f"  Initial alive: {initial_alive}")
            print(f"  Final alive: {final_alive}")
            print(f"  Health status: {health}")
            print(f"  Stability score: {stability_score:.4f}")
            print(f"  Steps completed: {len(alive_counts) - 1}/{scenario['steps']}")
    
    # Summary
    print("\n" + "=" * 50)
    print("ADVANCED HOMEOSTATIC NCA TEST SUMMARY")
    print("=" * 50)
    
    healthy_scenarios = 0
    stable_scenarios = 0
    
    for name, result in results.items():
        print(f"{name:20s}: {result['health']:10s} (alive: {result['final_alive']:4.0f}, stability: {result['stability_score']:.3f})")
        if result['health'] == "HEALTHY":
            healthy_scenarios += 1
        if result['stability_score'] > 0.1:
            stable_scenarios += 1
    
    print(f"\nHealthy scenarios: {healthy_scenarios}/{len(scenarios)}")
    print(f"Stable scenarios: {stable_scenarios}/{len(scenarios)}")
    
    # Overall assessment
    if healthy_scenarios >= len(scenarios) * 0.6:
        print("\n✅ ADVANCED HOMEOSTATIC SYSTEM SHOWS PROMISE")
        print("The attention-memory-hardware system demonstrates improved stability")
        print("This represents a significant advance over simple bifurcation dynamics")
    elif stable_scenarios >= len(scenarios) * 0.4:
        print("\n⚠️ PARTIAL SUCCESS")
        print("Some stability improvements but bifurcation still present")
        print("The advanced mechanisms show potential but need refinement")
    else:
        print("\n❌ ADVANCED HOMEOSTATIC SYSTEM STILL UNSTABLE")
        print("Bifurcation problem persists despite advanced mechanisms")
        print("May need even more sophisticated regulation or different approach")
    
    # Technical analysis
    print("\n" + "=" * 50)
    print("TECHNICAL ANALYSIS")
    print("=" * 50)
    
    # Analyze patterns in the results
    death_scenarios = sum(1 for r in results.values() if r['health'] == 'DEAD')
    overgrowth_scenarios = sum(1 for r in results.values() if r['health'] == 'OVERGROWN')
    
    print(f"Death scenarios: {death_scenarios}")
    print(f"Overgrowth scenarios: {overgrowth_scenarios}")
    print(f"Stable scenarios: {stable_scenarios}")
    
    if death_scenarios > overgrowth_scenarios:
        print("\n⚠️ DEATH-BIASED SYSTEM")
        print("The homeostatic controller may be too aggressive in reducing population")
        print("Recommendation: Reduce death promotion in hardware configuration")
    elif overgrowth_scenarios > death_scenarios:
        print("\n⚠️ GROWTH-BIASED SYSTEM")
        print("The homeostatic controller may not be strong enough to prevent overgrowth")
        print("Recommendation: Strengthen population regulation mechanisms")
    else:
        print("\n✅ BALANCED DYNAMICS")
        print("The system shows balanced behavior between death and overgrowth")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"homeostatic_nca_test_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for name, result in results.items():
        json_results[name] = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                             for k, v in result.items()}
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    results = test_homeostatic_nca() 