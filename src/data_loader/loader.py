import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

if __name__ == '__main__':
    # A simple test to verify the data loader
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ukiyo-e')
    dataset = ImageDataset(data_dir)
    print(f"Found {len(dataset)} images.")
    
    if len(dataset) > 0:
        img = dataset[0]
        print(f"Shape of a single image tensor: {img.shape}")
        print(f"Data type: {img.dtype}")
        print(f"Min value: {img.min()}, Max value: {img.max()}")
        
        # To visualize the image, we need to denormalize it
        import matplotlib.pyplot as plt
        import numpy as np
        
        img_denorm = img / 2 + 0.5 # Denormalize
        plt.imshow(np.transpose(img_denorm.numpy(), (1, 2, 0)))
        plt.title("Sample Image from Dataset")
        plt.show() 