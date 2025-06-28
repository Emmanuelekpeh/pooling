#!/usr/bin/env python3
"""
Create a smaller dataset for faster training iteration
"""
import os
import shutil
import random

def create_small_dataset(source_dir="./data/ukiyo-e", target_dir="./data/ukiyo-e-small", num_images=20):
    """Create a smaller dataset by copying a subset of images"""
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} total images")
    
    # Randomly select images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    # Copy selected images
    for img in selected_images:
        src_path = os.path.join(source_dir, img)
        dst_path = os.path.join(target_dir, img)
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {img}")
    
    print(f"\nCreated small dataset with {len(selected_images)} images in {target_dir}")
    return target_dir

if __name__ == "__main__":
    create_small_dataset() 