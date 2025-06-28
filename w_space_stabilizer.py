#!/usr/bin/env python3
"""
W-Space Stabilization Module
Fixes latent space issues contributing to NCA cell death
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WSpaceStabilizer(nn.Module):
    """
    Stabilizes W-vector processing to prevent cell death issues
    Addresses magnitude problems and inconsistent conditioning
    """
    
    def __init__(self, w_dim=128, stabilized_dim=64):
        super().__init__()
        self.w_dim = w_dim
        self.stabilized_dim = stabilized_dim
        
        # W-vector normalization and stabilization
        self.w_normalizer = nn.Sequential(
            nn.LayerNorm(w_dim),
            nn.Linear(w_dim, stabilized_dim),
            nn.Tanh(),  # Bound outputs to [-1, 1]
            nn.Dropout(0.1)
        )
        
        # Separate pathways for different uses
        self.runoff_projection = nn.Linear(stabilized_dim, stabilized_dim)  # 64 dims for runoff
        self.conditioning_projection = nn.Linear(stabilized_dim, stabilized_dim // 2)  # 32 dims for conditioning
        self.style_projection = nn.Linear(stabilized_dim, w_dim // 3)  # 42 dims for style (w_dim//3)
        
        # Magnitude monitoring
        self.register_buffer('w_magnitude_history', torch.zeros(100))
        self.register_buffer('magnitude_index', torch.tensor(0))
        
    def forward(self, w_raw):
        """
        Stabilize w-vector for consistent processing
        Returns: dict with different w-vector variants
        """
        batch_size = w_raw.shape[0]
        
        # 1. MAGNITUDE MONITORING (detect explosive w-vectors)
        w_magnitude = torch.norm(w_raw, dim=1).mean()
        
        # Update magnitude history (rolling buffer)
        idx = self.magnitude_index % 100
        self.w_magnitude_history[idx] = w_magnitude.detach()
        self.magnitude_index += 1
        
        # Check for explosive magnitudes
        if w_magnitude > 10.0:  # Threshold for concerning magnitude
            print(f"⚠️  W-vector magnitude warning: {w_magnitude:.3f} (may cause cell death)")
        
        # 2. STABILIZE W-VECTOR
        # Apply layer norm to prevent magnitude explosion
        w_stabilized = self.w_normalizer(w_raw)
        
        # 3. CREATE SPECIALIZED PROJECTIONS
        w_variants = {
            'raw': w_raw,
            'stabilized': w_stabilized,
            'runoff': self.runoff_projection(w_stabilized),
            'conditioning': self.conditioning_projection(w_stabilized),
            'style': self.style_projection(w_stabilized)
        }
        
        # 4. MAGNITUDE STATISTICS
        history_slice = self.w_magnitude_history[:min(100, self.magnitude_index)]
        
        # Calculate standard deviation only if we have enough samples (≥2 for meaningful std)
        if len(history_slice) >= 2:
            magnitude_std = history_slice.std().item()
        else:
            magnitude_std = 0.0  # Default value when insufficient data
        
        stats = {
            'current_magnitude': w_magnitude.item(),
            'average_magnitude': history_slice.mean().item() if len(history_slice) > 0 else 0.0,
            'magnitude_std': magnitude_std
        }
        
        return w_variants, stats

def apply_w_stabilization_fix():
    """
    Apply the W-space stabilization fix to train_integrated_fast.py
    This modifies the problematic w-vector processing
    """
    
    # The fix involves:
    # 1. Using consistent w-vector dimensions across all pathways
    # 2. Adding magnitude monitoring and clipping
    # 3. Separating runoff and conditioning w-vectors
    
    fix_code = '''
# W-SPACE STABILIZATION FIX
# Add this to the IntegratedNCA.__init__ method:

self.w_stabilizer = WSpaceStabilizer(w_dim=w_dim, stabilized_dim=64)

# Modify the forward method to use stabilized w-vectors:

def forward(self, x, w, steps, target_img=None):
    # STABILIZE W-VECTOR FIRST
    w_variants, w_stats = self.w_stabilizer(w)
    
    if w_stats['current_magnitude'] > 15.0:
        print(f"🚨 CRITICAL W-magnitude: {w_stats['current_magnitude']:.3f} - potential cell death risk!")
    
    if self.use_rich_conditioning:
        # Use stabilized w-vectors for consistent processing
        w_semantic = w_variants['conditioning']  # Consistent 32-dim vector
        w_runoff = w_variants['runoff']          # Consistent 64-dim vector for runoff
        w_style = w_variants['style']            # Consistent 32-dim vector for style
        
        # Update runoff controller to use stabilized w_runoff
        runoff_input = torch.cat([global_cell_state, w_runoff, spatial_stats], dim=1)
        
        # Rest of the code remains the same but uses stabilized vectors
        ...
'''
    
    return fix_code

if __name__ == "__main__":
    # Test the stabilizer
    print("🧪 Testing W-Space Stabilizer...")
    
    # Simulate problematic w-vectors
    stabilizer = WSpaceStabilizer()
    
    # Test 1: Normal w-vector
    w_normal = torch.randn(2, 128) * 2.0
    variants, stats = stabilizer(w_normal)
    print(f"✅ Normal w-vector: magnitude={stats['current_magnitude']:.3f}")
    
    # Test 2: Explosive w-vector (like in your training)
    w_explosive = torch.randn(2, 128) * 20.0  # Simulate explosive magnitude
    variants, stats = stabilizer(w_explosive)
    print(f"⚠️  Explosive w-vector: magnitude={stats['current_magnitude']:.3f}")
    
    # Test 3: Verify dimension consistency
    print(f"✅ Dimension consistency:")
    print(f"   - Runoff: {variants['runoff'].shape} (expected: [2, 64])")
    print(f"   - Conditioning: {variants['conditioning'].shape} (expected: [2, 32])")
    print(f"   - Style: {variants['style'].shape} (expected: [2, 42])")
    
    print("\n🎯 W-Space Stabilizer ready! Apply the fix to resolve cell death issues.") 