import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated import IntegratedGenerator, Z_DIM, W_DIM, DEVICE

def test_generator():
    """Test if the generator produces varied outputs"""
    print("Testing StyleGAN Generator...")
    
    # Create generator
    generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
    generator.eval()
    
    # Generate multiple samples
    with torch.no_grad():
        # Test with different noise inputs
        noise1 = torch.randn(1, Z_DIM).to(DEVICE)
        noise2 = torch.randn(1, Z_DIM).to(DEVICE)
        noise3 = torch.randn(1, Z_DIM).to(DEVICE)
        
        img1, w1 = generator(noise1, return_w=True)
        img2, w2 = generator(noise2, return_w=True)
        img3, w3 = generator(noise3, return_w=True)
        
        # Check if outputs are different
        diff_12 = F.mse_loss(img1, img2).item()
        diff_13 = F.mse_loss(img1, img3).item()
        diff_23 = F.mse_loss(img2, img3).item()
        
        print(f"MSE differences between generated images:")
        print(f"  Image 1 vs 2: {diff_12:.6f}")
        print(f"  Image 1 vs 3: {diff_13:.6f}")
        print(f"  Image 2 vs 3: {diff_23:.6f}")
        
        # Check value ranges
        print(f"\nImage value ranges:")
        print(f"  Image 1: [{img1.min().item():.3f}, {img1.max().item():.3f}]")
        print(f"  Image 2: [{img2.min().item():.3f}, {img2.max().item():.3f}]")
        print(f"  Image 3: [{img3.min().item():.3f}, {img3.max().item():.3f}]")
        
        # Check w vector differences
        w_diff_12 = F.mse_loss(w1, w2).item()
        w_diff_13 = F.mse_loss(w1, w3).item()
        w_diff_23 = F.mse_loss(w2, w3).item()
        
        print(f"\nW vector differences:")
        print(f"  W1 vs W2: {w_diff_12:.6f}")
        print(f"  W1 vs W3: {w_diff_13:.6f}")
        print(f"  W2 vs W3: {w_diff_23:.6f}")
        
        # Save sample images
        def save_tensor_as_image(tensor, filename):
            # Convert from [-1, 1] to [0, 1]
            img_np = (tensor.squeeze(0).cpu().numpy() + 1.0) / 2.0
            img_np = np.clip(img_np.transpose(1, 2, 0), 0, 1)
            plt.figure(figsize=(4, 4))
            plt.imshow(img_np)
            plt.axis('off')
            plt.title(f'Generated Image ({filename})')
            plt.savefig(f'test_{filename}.png', bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved test_{filename}.png")
        
        save_tensor_as_image(img1, 'gen1')
        save_tensor_as_image(img2, 'gen2')
        save_tensor_as_image(img3, 'gen3')
        
        # Assessment
        if diff_12 > 0.01 and diff_13 > 0.01 and diff_23 > 0.01:
            print("\n✅ Generator appears to be working - producing varied outputs")
        else:
            print("\n❌ Generator may be stuck - outputs are too similar")
            
        if abs(img1.mean().item()) < 0.1:
            print("⚠️  Generator outputs are close to zero - may need better initialization")
        
        return img1, img2, img3

if __name__ == "__main__":
    test_generator() 