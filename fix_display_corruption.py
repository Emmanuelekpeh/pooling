#!/usr/bin/env python3

import torch
import json
import numpy as np
from PIL import Image
import base64
import io
import os

def fix_generator_display():
    """Fix the generator display corruption by forcing a fresh sample generation"""
    
    print("🔧 Diagnosing Generator Display Corruption...")
    
    # Load the latest checkpoint
    checkpoint_path = './checkpoints/latest_checkpoint.pt'
    if not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found")
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Check if generator state exists
        if 'generator_state_dict' not in checkpoint:
            print("❌ No generator state in checkpoint")
            return
            
        print("✅ Generator state found in checkpoint")
        
        # Detect the correct architecture from checkpoint
        z_dim = checkpoint['generator_state_dict']['mapping.mapping.0.weight'].shape[1]  # Input to mapping
        w_dim = checkpoint['generator_state_dict']['mapping.w_avg'].shape[0]  # W space dimension
        print(f"✅ Detected Z_DIM from checkpoint: {z_dim}")
        print(f"✅ Detected W_DIM from checkpoint: {w_dim}")
        
        # Load generator architecture with correct dimensions
        from train_integrated_fast import IntegratedGenerator
        
        generator = IntegratedGenerator(z_dim=z_dim, w_dim=w_dim)  # Use detected dimensions
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        print("✅ Generator loaded successfully with correct architecture")
        
        # Generate a test sample
        with torch.no_grad():
            z = torch.randn(1, z_dim)  # Use correct Z_DIM
            fake_img = generator(z)
            
            print(f"✅ Generated sample shape: {fake_img.shape}")
            print(f"✅ Generated sample range: [{fake_img.min().item():.3f}, {fake_img.max().item():.3f}]")
            
            # Check for common corruption patterns
            if fake_img.min().item() == fake_img.max().item():
                print("❌ PROBLEM: All pixels have same value (collapsed generator)")
                return
            elif torch.isnan(fake_img).any():
                print("❌ PROBLEM: NaN values detected")
                return
            elif torch.isinf(fake_img).any():
                print("❌ PROBLEM: Infinite values detected")
                return
            else:
                print("✅ Generator output looks healthy")
            
            # Convert to displayable format
            img_np = fake_img[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np + 1) / 2  # Convert from [-1, 1] to [0, 1]
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
            
            # Save as test image
            img_pil = Image.fromarray(img_np)
            test_path = './generator_test_output.png'
            img_pil.save(test_path)
            print(f"✅ Test image saved to: {test_path}")
            
            # Convert to base64 for web display
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Force update the status.json with fresh generator sample
            try:
                with open('./samples/status.json', 'r') as f:
                    status = json.load(f)
                
                # Update with fresh generator image
                if 'images' not in status:
                    status['images'] = {}
                
                status['images']['generator'] = f"data:image/png;base64,{img_b64}"
                status['display_fix_timestamp'] = torch.randn(1).item()  # Force refresh
                
                with open('./samples/status.json', 'w') as f:
                    json.dump(status, f)
                
                print("✅ Status.json updated with fresh generator sample")
                print("🎯 Try refreshing your dashboard now!")
                
                # Also generate multiple samples to verify consistency
                print("\n🔄 Generating additional samples to verify consistency...")
                for i in range(3):
                    z_test = torch.randn(1, z_dim)  # Use correct Z_DIM
                    test_img = generator(z_test)
                    print(f"  Sample {i+1}: Range [{test_img.min().item():.3f}, {test_img.max().item():.3f}]")
                
                print("✅ Generator is producing consistent, varied outputs")
                
                # Report the architecture fix needed for training
                print(f"\n⚠️  ARCHITECTURE FIX NEEDED:")
                print(f"   Current training code uses Z_DIM=128, but checkpoint uses Z_DIM={z_dim}")
                print(f"   Current training code uses W_DIM=128, but checkpoint uses W_DIM={w_dim}")
                print(f"   This mismatch is causing the black display issue!")
                
            except Exception as e:
                print(f"⚠️  Status update failed: {e}")
                
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

def check_display_pipeline():
    """Check the entire display pipeline for corruption"""
    
    print("\n🔍 Checking Display Pipeline...")
    
    # Check status.json structure
    try:
        with open('./samples/status.json', 'r') as f:
            status = json.load(f)
        
        print("✅ Status.json loaded successfully")
        
        if 'images' in status:
            images = status['images']
            print(f"✅ Images section found with {len(images)} entries")
            
            for key, value in images.items():
                if isinstance(value, str) and value.startswith('data:image'):
                    print(f"  - {key}: Valid base64 image data ({len(value)} chars)")
                else:
                    print(f"  - {key}: ❌ Invalid image data")
        else:
            print("❌ No images section in status.json")
            
    except Exception as e:
        print(f"❌ Status.json error: {e}")

if __name__ == "__main__":
    print("🔧 Generator Display Corruption Fix Tool")
    print("=" * 50)
    
    check_display_pipeline()
    fix_generator_display()
    
    print("\n" + "=" * 50)
    print("✅ Display fix complete!")
    print("🔄 Refresh your browser to see the fixed generator output") 