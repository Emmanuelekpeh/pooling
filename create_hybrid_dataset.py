#!/usr/bin/env python3
"""
Hybrid Dataset Creator for Optimal NCA Training

Implements user's brilliant 70/30 strategy:
- 70% Traditional Ukiyo-e: Preserves flowing lines, gradients, traditional art elements
- 30% Pixel Art Optimized: Maintains NCA-friendly training characteristics

This balanced approach ensures the NCA learns both:
1. Smooth, organic flowing lines (traditional Ukiyo-e strength)
2. Stable growth patterns (pixel art optimization strength)
"""

import os
import random
import shutil
from pathlib import Path
import argparse
from PIL import Image
import numpy as np

def count_images_in_dir(directory):
    """Count image files in a directory"""
    if not os.path.exists(directory):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    count = 0
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in image_extensions):
            count += 1
    return count

def get_image_files(directory):
    """Get list of image files from directory"""
    if not os.path.exists(directory):
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    files = []
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in image_extensions):
            files.append(file)
    return sorted(files)

def analyze_current_datasets():
    """Analyze existing datasets to understand what we're working with"""
    original_dir = Path("./data/ukiyo-e")  # Use full ukiyo-e dataset
    pixelart_dir = Path("./data/ukiyo-e-pixelart")
    
    print("📊 Current Dataset Analysis:")
    
    original_count = count_images_in_dir(original_dir)
    pixelart_count = count_images_in_dir(pixelart_dir)
    
    print(f"   Traditional Ukiyo-e: {original_count} images")
    print(f"   Pixel Art Optimized: {pixelart_count} images")
    
    return original_count, pixelart_count

def create_hybrid_dataset(traditional_ratio=0.7, target_total=40, create_comparisons=False):
    """
    Create hybrid dataset using user's brilliant 70/30 strategy
    Limited to same total as before (40 images) but from full ukiyo-e dataset
    """
    print("🎨 Creating Hybrid Dataset (User's 70/30 Strategy)")
    print(f"   Traditional ratio: {traditional_ratio*100:.0f}%")
    print(f"   Pixel art ratio: {(1-traditional_ratio)*100:.0f}%")
    print(f"   Target total images: {target_total}")
    
    # Source directories
    original_dir = Path("./data/ukiyo-e")  # Full ukiyo-e dataset
    pixelart_dir = Path("./data/ukiyo-e-pixelart")
    
    # Output directory
    output_dir = Path("./data/ukiyo-e-hybrid")
    output_dir.mkdir(exist_ok=True)
    
    # Clear existing files
    for file in output_dir.glob("*"):
        if file.is_file():
            file.unlink()
    
    # Get available files
    original_files = get_image_files(original_dir)
    pixelart_files = get_image_files(pixelart_dir)
    
    print(f"   Available traditional: {len(original_files)}")
    print(f"   Available pixel art: {len(pixelart_files)}")
    
    # Calculate target counts
    target_traditional = int(target_total * traditional_ratio)
    target_pixelart = target_total - target_traditional
    
    print(f"   Target traditional: {target_traditional}")
    print(f"   Target pixel art: {target_pixelart}")
    
    # Randomly sample from full dataset to get variety
    random.seed(42)  # For reproducible results
    selected_traditional = random.sample(original_files, min(target_traditional, len(original_files)))
    selected_pixelart = random.sample(pixelart_files, min(target_pixelart, len(pixelart_files)))
    
    print(f"\n📁 Copying traditional Ukiyo-e images...")
    copied_traditional = 0
    for i, filename in enumerate(selected_traditional):
        src = original_dir / filename
        dst = output_dir / filename
        shutil.copy2(src, dst)
        copied_traditional += 1
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(selected_traditional)} traditional images")
    
    print(f"\n🎮 Copying pixel art optimized images...")
    copied_pixelart = 0
    for i, filename in enumerate(selected_pixelart):
        src = pixelart_dir / filename
        dst = output_dir / filename
        shutil.copy2(src, dst)
        copied_pixelart += 1
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(selected_pixelart)} pixel art images")
    
    total_copied = copied_traditional + copied_pixelart
    actual_traditional_ratio = copied_traditional / total_copied if total_copied > 0 else 0
    
    # Save composition details
    composition_file = output_dir / "dataset_composition.txt"
    with open(composition_file, 'w', encoding='utf-8') as f:
        f.write(f"Hybrid Dataset Composition\n")
        f.write(f"=========================\n\n")
        f.write(f"Created with user's brilliant 70/30 strategy\n")
        f.write(f"Total images: {total_copied}\n")
        f.write(f"Traditional Ukiyo-e: {copied_traditional} ({actual_traditional_ratio*100:.1f}%)\n")
        f.write(f"Pixel Art Optimized: {copied_pixelart} ({(1-actual_traditional_ratio)*100:.1f}%)\n\n")
        f.write(f"Traditional Images:\n")
        for filename in selected_traditional:
            f.write(f"  - {filename}\n")
        f.write(f"\nPixel Art Images:\n")
        for filename in selected_pixelart:
            f.write(f"  - {filename}\n")
    
    print(f"\n✅ Hybrid dataset created successfully!")
    print(f"   Total images: {total_copied}")
    print(f"   Traditional: {copied_traditional} ({actual_traditional_ratio*100:.1f}%)")
    print(f"   Pixel art: {copied_pixelart} ({(1-actual_traditional_ratio)*100:.1f}%)")
    print(f"   Output directory: {output_dir}")
    print(f"   Composition details: {composition_file}")
    
    # Create comparison samples if requested
    if create_comparisons:
        create_comparison_samples(selected_traditional[:3], selected_pixelart[:3], original_dir, pixelart_dir)
    
    # Save training configuration update
    config_file = "hybrid_training_update.txt"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(f"# Hybrid Dataset Training Configuration\n")
        f.write(f"# Update train_integrated_fast.py with:\n\n")
        f.write(f'DATA_DIR = "./data/ukiyo-e-hybrid"\n')
        f.write(f"# Total images: {total_copied}\n")
        f.write(f"# Traditional: {copied_traditional} ({actual_traditional_ratio*100:.1f}%)\n")
        f.write(f"# Pixel art: {copied_pixelart} ({(1-actual_traditional_ratio)*100:.1f}%)\n\n")
        f.write(f"# Expected benefits:\n")
        f.write(f"# - Stable NCA growth from pixel art component\n")
        f.write(f"# - Rich traditional art learning from ukiyo-e component\n")
        f.write(f"# - Best of both worlds approach\n")
    
    return output_dir, total_copied, actual_traditional_ratio

def create_comparison_samples(traditional_files, pixelart_files, original_dir, pixelart_dir):
    """Create side-by-side comparison samples"""
    comparison_dir = Path("./data/hybrid-comparison")
    comparison_dir.mkdir(exist_ok=True)
    
    print(f"📊 Creating comparison samples...")
    
    # Copy a few examples for comparison
    for i, (trad_file, pixel_file) in enumerate(zip(traditional_files, pixelart_files)):
        # Copy traditional
        trad_src = original_dir / trad_file
        trad_dst = comparison_dir / f"traditional_{i+1}_{trad_file}"
        shutil.copy2(trad_src, trad_dst)
        
        # Copy pixel art
        pixel_src = pixelart_dir / pixel_file
        pixel_dst = comparison_dir / f"pixelart_{i+1}_{pixel_file}"
        shutil.copy2(pixel_src, pixel_dst)
    
    print(f"📊 Comparison samples saved to: {comparison_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create hybrid ukiyo-e dataset')
    parser.add_argument('--traditional-ratio', type=float, default=0.7,
                       help='Ratio of traditional images (default: 0.7 for 70%)')
    parser.add_argument('--total-images', type=int, default=40,
                       help='Total number of images in hybrid dataset (default: 40)')
    parser.add_argument('--create-comparisons', action='store_true',
                       help='Create comparison samples')
    
    args = parser.parse_args()
    
    print("🎨 Hybrid Dataset Creator")
    print("   Implementing user's brilliant 70/30 strategy!")
    print("   70% Traditional: Flowing lines & gradients")
    print("   30% Pixel Art: NCA-friendly growth patterns")
    print(f"   Sampling from FULL ukiyo-e dataset ({count_images_in_dir('./data/ukiyo-e')} images)")
    
    # Analyze current datasets
    analyze_current_datasets()
    
    # Create hybrid dataset
    output_dir, total_images, actual_ratio = create_hybrid_dataset(
        traditional_ratio=args.traditional_ratio,
        target_total=args.total_images,
        create_comparisons=args.create_comparisons
    )
    
    print(f"📝 Training configuration update saved to: hybrid_training_update.txt")
    
    print(f"\n🚀 Ready for enhanced training!")
    print(f"   Update DATA_DIR in train_integrated_fast.py to: {output_dir}")
    print(f"   Expected: Best of both worlds - stable NCA growth + flowing art learning")

if __name__ == "__main__":
    main() 