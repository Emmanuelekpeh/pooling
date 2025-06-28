#!/usr/bin/env python3
"""
Quick test of the consolidated training approach
"""

import torch
import sys

print("🧪 Testing Consolidated Training Approach")
print("=" * 50)

try:
    # Test imports from train_integrated_fast.py
    from train_integrated_fast import (
        Z_DIM, W_DIM, IMG_SIZE, DEVICE,
        IntegratedGenerator, IntegratedNCA, Discriminator
    )
    print("✅ Successfully imported core models")
    
    # Test model initialization
    generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
    nca = IntegratedNCA(8, W_DIM).to(DEVICE)  # Use 8 channels
    discriminator = Discriminator(IMG_SIZE).to(DEVICE)
    print("✅ Successfully initialized models")
    
    # Test forward pass
    z = torch.randn(2, Z_DIM, device=DEVICE)
    fake_imgs, w = generator(z, return_w=True)
    print(f"✅ Generator output: {fake_imgs.shape}, W: {w.shape}")
    
    # Test NCA with proper seed
    nca_seed = nca.get_seed(2, IMG_SIZE, DEVICE)
    nca_output = nca(nca_seed, w, 20)
    nca_imgs = nca.to_rgb(nca_output)
    print(f"✅ NCA output: {nca_imgs.shape}")
    
    # Test discriminator
    disc_output = discriminator(fake_imgs)
    print(f"✅ Discriminator output: {disc_output.shape}")
    
    print("\n🎉 All tests passed! Consolidated approach works!")
    print("The consolidation successfully uses existing models.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 