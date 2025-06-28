#!/usr/bin/env python3
"""
Pixel Art Dataset Preprocessor for Ukiyo-e Images

This script implements the user's brilliant insight: pixel art aesthetics work much better 
at 64x64 resolution than trying to preserve fine details from traditional art.

The preprocessing pipeline:
1. Reduces color palette to 8-16 colors (pixel art style)
2. Applies geometric simplification while preserving composition
3. Enhances contrast and edges for better NCA training
4. Creates clean, training-optimized 64x64 versions
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def reduce_colors_kmeans(image, n_colors=12):
    """Reduce image colors using K-means clustering (pixel art style)"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    original_shape = img_array.shape
    
    # Reshape for K-means
    img_flat = img_array.reshape(-1, 3)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(img_flat)
    
    # Replace pixels with cluster centers
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    new_img = new_colors.reshape(original_shape).astype(np.uint8)
    
    return Image.fromarray(new_img)

def enhance_contrast_and_edges(image, contrast_factor=1.3, edge_enhance=True):
    """Enhance contrast and edges for better pixel art appearance"""
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    
    # Enhance sharpness for pixel art crispness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    # Optional edge enhancement
    if edge_enhance:
        image = image.filter(ImageFilter.EDGE_ENHANCE)
    
    return image

def apply_geometric_simplification(image, blur_radius=0.8):
    """Apply slight blur to reduce fine details while preserving main shapes"""
    # Convert to numpy for OpenCV processing
    img_array = np.array(image)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(img_array, 9, 75, 75)
    
    # Light Gaussian blur for simplification
    blurred = cv2.GaussianBlur(filtered, (3, 3), blur_radius)
    
    return Image.fromarray(blurred)

def create_pixel_art_version(image_path, output_path, n_colors=12, size=(64, 64)):
    """
    Create a pixel-art optimized version of an image
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        n_colors: Number of colors in final palette (8-16 works well)
        size: Output size (64x64 for current training)
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Step 1: Resize to target size with good resampling
        image = image.resize(size, Image.Resampling.LANCZOS)
        
        # Step 2: Apply geometric simplification
        image = apply_geometric_simplification(image, blur_radius=0.6)
        
        # Step 3: Reduce color palette (key pixel art characteristic)
        image = reduce_colors_kmeans(image, n_colors=n_colors)
        
        # Step 4: Enhance contrast and edges for pixel art crispness
        image = enhance_contrast_and_edges(image, contrast_factor=1.4, edge_enhance=True)
        
        # Step 5: Final resize to ensure exact dimensions
        image = image.resize(size, Image.Resampling.NEAREST)  # Nearest neighbor for pixel art
        
        # Save processed image
        image.save(output_path, 'JPEG', quality=95)
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def analyze_color_distribution(image_path):
    """Analyze color distribution to determine optimal palette size"""
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    
    # Get unique colors
    unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
    
    return unique_colors

def create_comparison_grid(original_path, processed_path, save_path):
    """Create a before/after comparison image"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original
    original = Image.open(original_path)
    axes[0].imshow(original)
    axes[0].set_title("Original Ukiyo-e")
    axes[0].axis('off')
    
    # Processed
    processed = Image.open(processed_path)
    axes[1].imshow(processed)
    axes[1].set_title("Pixel Art Optimized")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_dataset(input_dir, output_dir, n_colors=12, create_comparisons=True):
    """
    Process entire dataset to create pixel art optimized versions
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save processed images
        n_colors: Number of colors in pixel art palette
        create_comparisons: Whether to create before/after comparison images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    comparison_path = output_path / "comparisons"
    
    # Create output directories
    output_path.mkdir(exist_ok=True)
    if create_comparisons:
        comparison_path.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix in image_extensions]
    
    print(f"🎨 Processing {len(image_files)} images for pixel art optimization...")
    print(f"   Color palette: {n_colors} colors")
    print(f"   Target size: 64x64")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    
    successful = 0
    failed = 0
    
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        # Create output filename
        output_file = output_path / f"pixelart_{image_file.stem}.jpg"
        
        # Process image
        success = create_pixel_art_version(
            str(image_file), 
            str(output_file), 
            n_colors=n_colors
        )
        
        if success:
            successful += 1
            
            # Create comparison if requested
            if create_comparisons:
                comparison_file = comparison_path / f"comparison_{image_file.stem}.png"
                create_comparison_grid(
                    str(image_file),
                    str(output_file),
                    str(comparison_file)
                )
        else:
            failed += 1
    
    print(f"\n✅ Processing complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful/(successful+failed)*100:.1f}%")
    
    if create_comparisons:
        print(f"   Comparison images saved to: {comparison_path}")
    
    return successful, failed

def main():
    parser = argparse.ArgumentParser(description="Create pixel art optimized dataset for NCA training")
    parser.add_argument("--input", "-i", default="./data/ukiyo-e-small", 
                       help="Input directory with original images")
    parser.add_argument("--output", "-o", default="./data/ukiyo-e-pixelart", 
                       help="Output directory for processed images")
    parser.add_argument("--colors", "-c", type=int, default=12, 
                       help="Number of colors in pixel art palette (8-16 recommended)")
    parser.add_argument("--no-comparisons", action="store_true", 
                       help="Skip creating before/after comparison images")
    
    args = parser.parse_args()
    
    print("🎨 Pixel Art Dataset Preprocessor")
    print("   Implementing user's brilliant insight: pixel art works better at 64x64!")
    print()
    
    # Validate input directory
    if not os.path.exists(args.input):
        print(f"❌ Error: Input directory '{args.input}' does not exist")
        return
    
    # Process dataset
    successful, failed = process_dataset(
        args.input, 
        args.output, 
        n_colors=args.colors,
        create_comparisons=not args.no_comparisons
    )
    
    if successful > 0:
        print(f"\n🚀 Ready for training!")
        print(f"   Update your training script to use: {args.output}")
        print(f"   Expected improvement: Better NCA learning due to pixel art optimization")
        print(f"   Recommended next step: Test with SimpleGrowthNCA")

if __name__ == "__main__":
    main() 