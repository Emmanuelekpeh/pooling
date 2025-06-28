#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

class StableNCA(nn.Module):
    """
    NCA with advanced stabilization mechanisms based on latest research:
    - Hardware-state separation
    - Private memory channels  
    - Multi-timescale dynamics
    - Differentiable stability control
    """
    def __init__(self, n_channels=16, n_memory=8, n_hardware=4, n_attention=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_memory = n_memory
        self.n_hardware = n_hardware
        self.n_attention = n_attention
        
        total_channels = n_channels + n_memory + n_hardware
        
        # Perception kernels
        self.register_buffer('identity', torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('laplacian', torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Perception network
        perception_size = total_channels * 4
        self.perception_net = nn.Sequential(
            nn.Linear(perception_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Multi-timescale attention mechanism
        self.fast_attention = nn.Linear(n_hardware, n_attention)
        self.slow_attention = nn.Linear(n_memory, n_attention)
        self.attention_temp = nn.Parameter(torch.tensor(1.0))
        
        # Multiple update pathways with different timescales
        self.fast_pathways = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_channels)
            ) for _ in range(n_attention)
        ])
        
        self.slow_pathways = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 16),
                nn.ReLU(), 
                nn.Linear(16, n_memory)
            ) for _ in range(n_attention)
        ])
        
        # Stability controller with predictive capability
        self.stability_predictor = nn.Sequential(
            nn.Linear(1 + 3, 32),  # memory_mean + alive_ratio + growth_rate + variance
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # growth_scale, death_scale, stability_scale
        )
        
        # Initialize with small random weights (not zeros)
        for pathway in self.fast_pathways:
            nn.init.normal_(pathway[-1].weight, 0, 0.01)
            nn.init.zeros_(pathway[-1].bias)
        for pathway in self.slow_pathways:
            nn.init.normal_(pathway[-1].weight, 0, 0.01)
            nn.init.zeros_(pathway[-1].bias)

    def perceive(self, x):
        batch_size, channels, height, width = x.shape
        perceived = []
        for kernel in [self.identity, self.sobel_x, self.sobel_y, self.laplacian]:
            kernel_expanded = kernel.expand(channels, -1, -1, -1)
            conv_result = F.conv2d(x, kernel_expanded, groups=channels, padding=1)
            perceived.append(conv_result)
        return torch.cat(perceived, dim=1)

    def forward(self, x, step_prob=0.5, stability_target=0.3):
        batch_size, total_channels, height, width = x.shape
        
        # Split channels
        visible = x[:, :self.n_channels]
        memory = x[:, self.n_channels:self.n_channels + self.n_memory]
        hardware = x[:, -self.n_hardware:]
        
        # Calculate population metrics
        alive_mask = (visible[:, 3:4] > 0.1).float()
        alive_ratio = alive_mask.mean(dim=[2, 3], keepdim=True)
        
        # Calculate growth rate and variance for predictive control
        if hasattr(self, 'prev_alive_ratio'):
            growth_rate = alive_ratio - self.prev_alive_ratio
            variance = alive_mask.var(dim=[2, 3], keepdim=True)
        else:
            growth_rate = torch.zeros_like(alive_ratio)
            variance = torch.zeros_like(alive_ratio)
        self.prev_alive_ratio = alive_ratio.detach()
        
        # Perception
        perception = self.perceive(x)
        perception_flat = perception.view(batch_size, -1, height * width).permute(0, 2, 1)
        memory_flat = memory.view(batch_size, -1, height * width).permute(0, 2, 1)
        hardware_flat = hardware.view(batch_size, -1, height * width).permute(0, 2, 1)
        
        # Process perception
        processed_perception = self.perception_net(perception_flat)
        
        # Multi-timescale attention
        fast_attention_logits = self.fast_attention(hardware_flat) / self.attention_temp
        slow_attention_logits = self.slow_attention(memory_flat) / self.attention_temp
        
        fast_weights = F.softmax(fast_attention_logits, dim=-1)
        slow_weights = F.softmax(slow_attention_logits, dim=-1)
        
        # Apply pathways with different timescales
        fast_outputs = []
        for i, pathway in enumerate(self.fast_pathways):
            output = pathway(processed_perception)
            fast_outputs.append(output)
        fast_stack = torch.stack(fast_outputs, dim=-1)
        
        slow_outputs = []
        for i, pathway in enumerate(self.slow_pathways):
            output = pathway(processed_perception)
            slow_outputs.append(output)
        slow_stack = torch.stack(slow_outputs, dim=-1)
        
        # Weighted combination
        fast_update = (fast_stack * fast_weights.unsqueeze(-2)).sum(dim=-1)
        slow_update = (slow_stack * slow_weights.unsqueeze(-2)).sum(dim=-1)
        
        # Predictive stability control
        alive_ratio_expanded = alive_ratio.expand(-1, -1, height, width).view(batch_size, height * width, 1)
        growth_rate_expanded = growth_rate.expand(-1, -1, height, width).view(batch_size, height * width, 1)
        variance_expanded = variance.expand(-1, -1, height, width).view(batch_size, height * width, 1)
        memory_mean = memory_flat.mean(dim=-1, keepdim=True)
        
        stability_input = torch.cat([memory_mean, alive_ratio_expanded, growth_rate_expanded, variance_expanded], dim=-1)
        stability_scales = self.stability_predictor(stability_input)
        growth_scale = torch.sigmoid(stability_scales[:, :, 0:1])
        death_scale = torch.sigmoid(stability_scales[:, :, 1:2])
        stability_scale = torch.sigmoid(stability_scales[:, :, 2:3])
        
        # Apply stability scaling based on current population state
        target_ratio = torch.tensor(stability_target, device=x.device)
        if alive_ratio.mean() < target_ratio * 0.5:  # Too few alive
            fast_update = fast_update * growth_scale * 2.0
        elif alive_ratio.mean() > target_ratio * 2.0:  # Too many alive
            fast_update = fast_update * death_scale * 0.5
        else:  # In target range
            fast_update = fast_update * stability_scale
        
        # Reshape back to spatial dimensions
        fast_spatial = fast_update.permute(0, 2, 1).view(batch_size, self.n_channels, height, width)
        slow_spatial = slow_update.permute(0, 2, 1).view(batch_size, self.n_memory, height, width)
        
        # Stochastic updates with different probabilities for fast/slow
        if self.training:
            fast_mask = (torch.rand_like(fast_spatial[:, :1]) < step_prob).float()
            slow_mask = (torch.rand_like(slow_spatial[:, :1]) < step_prob * 0.1).float()  # 10x slower
        else:
            fast_mask = torch.ones_like(fast_spatial[:, :1])
            slow_mask = torch.ones_like(slow_spatial[:, :1])
        
        # Apply updates
        new_visible = visible + fast_spatial * fast_mask
        new_memory = memory + slow_spatial * slow_mask
        
        # Enforce alive mask
        alive_extended = F.max_pool2d(alive_mask, 3, stride=1, padding=1)
        new_visible = new_visible * alive_extended
        new_memory = new_memory * alive_extended
        
        # Recombine
        new_x = torch.cat([new_visible, new_memory, hardware], dim=1)
        
        return new_x

def create_stability_curriculum():
    """Create a curriculum of increasingly complex stability tasks"""
    curriculum = [
        {
            'name': 'simple_growth',
            'target_alive_ratio': 0.1,
            'max_steps': 50,
            'stability_weight': 1.0,
            'description': 'Learn basic growth from small seed'
        },
        {
            'name': 'controlled_growth', 
            'target_alive_ratio': 0.2,
            'max_steps': 100,
            'stability_weight': 2.0,
            'description': 'Learn to control growth rate'
        },
        {
            'name': 'stable_population',
            'target_alive_ratio': 0.3,
            'max_steps': 200,
            'stability_weight': 5.0,
            'description': 'Maintain stable population'
        },
        {
            'name': 'dynamic_stability',
            'target_alive_ratio': 0.3,
            'max_steps': 500,
            'stability_weight': 10.0,
            'description': 'Long-term dynamic stability'
        },
        {
            'name': 'robust_stability',
            'target_alive_ratio': 0.3,
            'max_steps': 1000,
            'stability_weight': 20.0,
            'description': 'Stability under perturbations'
        }
    ]
    return curriculum

def train_curriculum_stage(model, stage, device, size=64):
    """Train a single stage of the curriculum"""
    print(f"\nTraining stage: {stage['name']}")
    print(f"Description: {stage['description']}")
    print(f"Target alive ratio: {stage['target_alive_ratio']}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    losses = []
    alive_ratios = []
    stability_scores = []
    
    n_epochs = 200
    
    for epoch in range(n_epochs):
        model.train()
        
        # Initialize state with small random seed
        state = torch.zeros(1, model.n_channels + model.n_memory + model.n_hardware, size, size, device=device)
        
        # Add seed
        center = size // 2
        seed_size = max(1, int(stage['target_alive_ratio'] * 10))
        state[0, 3, center-seed_size:center+seed_size+1, center-seed_size:center+seed_size+1] = 1.0
        state[0, 4:8, center-seed_size:center+seed_size+1, center-seed_size:center+seed_size+1] = torch.randn(4, seed_size*2+1, seed_size*2+1, device=device) * 0.1
        
        # Add hardware configuration
        hardware = torch.zeros(model.n_hardware, size, size, device=device)
        hardware[0] = stage['target_alive_ratio']  # Target population level
        hardware[1] = 1.0 / stage['max_steps']     # Time scale hint
        hardware[2] = stage['stability_weight']    # Stability importance
        hardware[3] = 0.5                          # Balance parameter
        state[0, -model.n_hardware:] = hardware
        
        # Run simulation
        alive_history = []
        for step in range(stage['max_steps']):
            state = model(state, step_prob=0.5, stability_target=stage['target_alive_ratio'])
            
            alive_mask = (state[:, 3:4] > 0.1).float()
            alive_ratio = alive_mask.mean().item()
            alive_history.append(alive_ratio)
            
            # Early stopping if system dies
            if alive_ratio < 0.001:
                break
        
        # Calculate losses
        target_ratio = stage['target_alive_ratio']
        
        # Population loss - use final state directly
        final_state = state
        final_alive_mask = (final_state[:, 3:4] > 0.1).float()
        final_alive_ratio = final_alive_mask.mean()
        population_loss = F.mse_loss(final_alive_ratio, torch.tensor(target_ratio, device=device))
        
        # Simple regularization loss to encourage stable dynamics
        # Penalize extreme values
        visible_state = final_state[:, :model.n_channels]
        regularization_loss = torch.mean(torch.abs(visible_state)) * 0.01
        
        # Combined loss
        total_loss = population_loss + regularization_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(total_loss.item())
        alive_ratios.append(final_alive_ratio.item())
        
        if len(alive_history) > 10:
            stability_score = 1.0 / (1.0 + np.var(alive_history[-10:]))
        else:
            stability_score = 0.0
        stability_scores.append(stability_score)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}, "
                  f"Alive ratio = {final_alive_ratio.item():.4f}, "
                  f"Stability = {stability_score:.4f}")
    
    return {
        'losses': losses,
        'alive_ratios': alive_ratios,
        'stability_scores': stability_scores,
        'final_performance': {
            'alive_ratio': alive_ratios[-1],
            'stability_score': stability_scores[-1],
            'target_achieved': abs(alive_ratios[-1] - target_ratio) < 0.05
        }
    }

def train_stable_nca_curriculum():
    """Train NCA using curriculum learning for stability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = StableNCA(n_channels=16, n_memory=8, n_hardware=4, n_attention=4).to(device)
    
    # Get curriculum
    curriculum = create_stability_curriculum()
    
    # Training results
    all_results = {}
    
    print("Starting Curriculum-Based Stable NCA Training")
    print("=" * 60)
    
    for stage in curriculum:
        results = train_curriculum_stage(model, stage, device)
        all_results[stage['name']] = results
        
        # Check if stage passed
        if results['final_performance']['target_achieved']:
            print(f"✅ Stage {stage['name']} PASSED")
        else:
            print(f"❌ Stage {stage['name']} FAILED")
            print(f"   Target: {stage['target_alive_ratio']:.3f}, "
                  f"Achieved: {results['final_performance']['alive_ratio']:.3f}")
        
        # Save checkpoint after each stage
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'stage': stage['name'],
            'results': results
        }
        torch.save(checkpoint, f'stable_nca_checkpoint_{stage["name"]}.pt')
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING SUMMARY")
    print("=" * 60)
    
    passed_stages = 0
    for stage_name, results in all_results.items():
        status = "PASSED" if results['final_performance']['target_achieved'] else "FAILED"
        if status == "PASSED":
            passed_stages += 1
        print(f"{stage_name:20s}: {status} (stability: {results['final_performance']['stability_score']:.3f})")
    
    print(f"\nStages passed: {passed_stages}/{len(curriculum)}")
    
    if passed_stages >= len(curriculum) * 0.8:
        print("\n🎉 CURRICULUM TRAINING SUCCESSFUL!")
        print("The NCA has learned stable population dynamics")
    elif passed_stages >= len(curriculum) * 0.6:
        print("\n⚠️ PARTIAL SUCCESS")
        print("The NCA shows improvement but needs refinement")
    else:
        print("\n❌ CURRICULUM TRAINING FAILED")
        print("The NCA could not learn stable dynamics")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"stable_nca_curriculum_results_{timestamp}.json"
    
    # Convert tensors to lists for JSON
    json_results = {}
    for stage_name, results in all_results.items():
        json_results[stage_name] = {
            'losses': results['losses'],
            'alive_ratios': results['alive_ratios'],
            'stability_scores': results['stability_scores'],
            'final_performance': results['final_performance']
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return model, all_results

if __name__ == "__main__":
    model, results = train_stable_nca_curriculum() 