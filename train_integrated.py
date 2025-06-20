import os
import io
import json
import time
import base64
import random
import argparse
from threading import Thread

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from flask import Flask, jsonify, render_template_string

# --- Configuration ---
Z_DIM = 128
W_DIM = 128
IMG_SIZE = 32
NCA_CHANNELS = 16
BATCH_SIZE = 4 # Reduced batch size further
LR = 1e-4
EPOCHS = 100
NCA_STEPS_MIN = 64
NCA_STEPS_MAX = 96
DEVICE = "cpu" # Force CPU
DATA_DIR = "./data/ukiyo-e"
SAMPLES_DIR = "./samples"

# --- Models ---
# (Using the same simplified model definitions as before)
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, w_dim)
        )
    def forward(self, x):
        return self.mapping(x)

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class IntegratedGenerator(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping_network = MappingNetwork(z_dim, w_dim)
        self.const = nn.Parameter(torch.randn(1, 256, 4, 4)) # Reduced
        self.up_block1 = nn.Upsample(scale_factor=2)
        self.gen_block1 = GeneratorBlock(256, 128) # Reduced
        self.up_block2 = nn.Upsample(scale_factor=2)
        self.gen_block2 = GeneratorBlock(128, 64) # Reduced
        self.up_block3 = nn.Upsample(scale_factor=2)
        self.gen_block3 = GeneratorBlock(64, 32) # Reduced
        self.up_block4 = nn.Upsample(scale_factor=2)
        self.gen_block4 = GeneratorBlock(32, 16) # Reduced
        self.to_rgb = nn.Conv2d(16, 3, kernel_size=1) # Reduced

    def forward(self, noise, return_w=False):
        w = self.mapping_network(noise)
        x = self.const.repeat(noise.shape[0], 1, 1, 1)
        x = self.up_block1(x)
        x = self.gen_block1(x)
        x = self.up_block2(x)
        x = self.gen_block2(x)
        x = self.up_block3(x)
        x = self.gen_block3(x)
        x = self.up_block4(x)
        x = self.gen_block4(x)
        img = torch.tanh(self.to_rgb(x))
        if return_w:
            return img, w
        return img

class IntegratedNCA(nn.Module):
    def __init__(self, channel_n, w_dim, hidden_n=128):
        super().__init__()
        self.channel_n = channel_n
        self.w_dim = w_dim
        self.update_net = nn.Sequential(
            nn.Linear(channel_n * 3 + w_dim, hidden_n),
            nn.ReLU(),
            nn.Linear(hidden_n, channel_n),
        )
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(DEVICE)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(DEVICE)

    def get_seed(self, batch_size, size, device):
        return torch.zeros(batch_size, self.channel_n, size, size, device=device)

    def to_rgba(self, x):
        return x[:, :4, :, :]

    def perceive(self, x):
        grad_x = F.conv2d(x, self.sobel_x.repeat(self.channel_n, 1, 1, 1), padding=1, groups=self.channel_n)
        grad_y = F.conv2d(x, self.sobel_y.repeat(self.channel_n, 1, 1, 1), padding=1, groups=self.channel_n)
        return torch.cat((x, grad_x, grad_y), 1)

    def forward(self, x, w, steps):
        for _ in range(steps):
            pre_life_mask = (F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1).float()
            perceived = self.perceive(x)
            perceived = perceived.permute(0, 2, 3, 1).reshape(-1, self.channel_n * 3)
            
            w_expanded = w.unsqueeze(1).unsqueeze(1).repeat(1, x.shape[2], x.shape[3], 1)
            w_reshaped = w_expanded.reshape(-1, self.w_dim)
            
            update_input = torch.cat([perceived, w_reshaped], dim=1)
            ds = self.update_net(update_input)
            
            ds = ds.reshape(x.shape[0], x.shape[2], x.shape[3], self.channel_n).permute(0, 3, 1, 2)
            
            x = x + ds
            life_mask = (x[:, 3:4, :, :] > 0.1).float() * pre_life_mask
            x = x * life_mask
        return x

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), # Reduced
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1), # Reduced
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), # Reduced
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), # Reduced
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 0),
        )
    def forward(self, img):
        return self.model(img).view(-1)

# --- Dataset ---
class ImageDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NCA vs StyleGAN Training</title>
    <style>
        body { font-family: sans-serif; background-color: #282c34; color: #abb2bf; text-align: center; }
        .container { display: flex; justify-content: center; align-items: flex-start; gap: 20px; margin-top: 20px; }
        .column { display: flex; flex-direction: column; align-items: center; }
        h1, h2 { color: #61afef; }
        img { border: 2px solid #61afef; max-width: 400px; height: auto; }
        #status { margin-top: 20px; font-size: 1.2em; min-height: 50px; padding: 10px; border-radius: 5px; background-color: #3b4048; }
        .error { color: #e06c75; border: 1px solid #e06c75; }
    </style>
</head>
<body>
    <h1>NCA vs StyleGAN Training Status</h1>
    <div id="status">Connecting...</div>
    <div class="container">
        <div class="column">
            <h2>Generator Output</h2>
            <img id="generator-image" src="https://via.placeholder.com/400" alt="Generator Output">
        </div>
        <div class="column">
            <h2>NCA Output</h2>
            <img id="nca-image" src="https://via.placeholder.com/400" alt="NCA Output">
        </div>
    </div>
    <script>
        function pollStatus() {
            setInterval(() => {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        statusDiv.textContent = data.status || 'No status message.';
                        if (data.error) {
                            statusDiv.classList.add('error');
                        } else {
                            statusDiv.classList.remove('error');
                        }

                        if (data.images && data.images.length > 0) {
                            document.getElementById('generator-image').src = 'data:image/png;base64,' + data.images[0];
                            if (data.images.length > 1) {
                                document.getElementById('nca-image').src = 'data:image/png;base64,' + data.images[1];
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching status:', error);
                        document.getElementById('status').textContent = 'Error: Could not connect to the server.';
                        document.getElementById('status').classList.add('error');
                    });
            }, 3000);
        }
        document.addEventListener('DOMContentLoaded', pollStatus);
    </script>
</body>
</html>
"""

# --- File-based state for inter-process communication ---
STATUS_FILE = os.path.join(SAMPLES_DIR, 'status.json')

def tensor_to_b64(tensor):
    """Converts a batch of tensors to a single base64 encoded PNG image grid."""
    # Ensure tensor is on CPU and detached from the computation graph
    tensor = tensor.detach().cpu()
    # Create a grid of images
    grid = torchvision.utils.make_grid(tensor, normalize=True, scale_each=True, nrow=4)
    # Convert to PIL Image
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # Save to buffer
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def update_status(status_message, images=None, error=False):
    """Writes the current status and images to a JSON file."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    # The 'images' argument is expected to be a list of tensors
    b64_images = [tensor_to_b64(img_tensor) for img_tensor in images] if images else []
    with open(STATUS_FILE, 'w') as f:
        json.dump({
            'status': status_message,
            'images': b64_images,
            'error': error,
            'timestamp': time.time()
        }, f)

# --- Training Function (to be run by the 'worker' process) ---
def training_loop():
    """The main training loop."""
    try:
        update_status("Initializing models and data...")

        # --- Create separate models and optimizers ---
        generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
        nca_model = IntegratedNCA(channel_n=NCA_CHANNELS, w_dim=W_DIM).to(DEVICE)
        discriminator = Discriminator(IMG_SIZE).to(DEVICE)

        # Optimizer for the Generator's mapping network
        opt_gen = optim.Adam(generator.mapping_network.parameters(), lr=LR, betas=(0.5, 0.999))
        # Optimizer for the NCA's update network
        opt_nca = optim.Adam(nca_model.parameters(), lr=LR * 10, betas=(0.5, 0.999)) # NCA often needs a higher LR
        # Optimizer for the Discriminator
        opt_disc = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
        
        dataset = ImageDataset(DATA_DIR, img_size=IMG_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

        fixed_noise = torch.randn(min(BATCH_SIZE, 8), Z_DIM).to(DEVICE)
        
        update_status("Starting training...")
        
        for epoch in range(EPOCHS):
            progress_bar = tqdm(dataloader, desc=f"E:{epoch+1}", leave=False)
            for batch_idx, real_imgs in enumerate(progress_bar):
                real_imgs = real_imgs.to(DEVICE)
                
                # Generate w vector from noise
                noise = torch.randn(real_imgs.shape[0], Z_DIM).to(DEVICE)
                
                # --- Train Discriminator ---
                discriminator.zero_grad()
                
                # On real images
                disc_real_pred = discriminator(real_imgs)
                loss_disc_real = F.binary_cross_entropy_with_logits(disc_real_pred, torch.ones_like(disc_real_pred))
                
                # On fake images from Generator
                with torch.no_grad():
                    fake_imgs_gen, w = generator(noise, return_w=True)
                disc_fake_gen_pred = discriminator(fake_imgs_gen.detach())
                loss_disc_fake_gen = F.binary_cross_entropy_with_logits(disc_fake_gen_pred, torch.zeros_like(disc_fake_gen_pred))

                # On fake images from NCA
                with torch.no_grad():
                    seed = nca_model.get_seed(batch_size=real_imgs.shape[0], size=IMG_SIZE, device=DEVICE)
                    nca_steps = random.randint(NCA_STEPS_MIN, NCA_STEPS_MAX)
                    nca_output_grid = nca_model(seed, w.detach(), steps=nca_steps)
                    fake_imgs_nca = nca_model.to_rgba(nca_output_grid)[:, :3, :, :] # Get RGB channels
                disc_fake_nca_pred = discriminator(fake_imgs_nca.detach())
                loss_disc_fake_nca = F.binary_cross_entropy_with_logits(disc_fake_nca_pred, torch.zeros_like(disc_fake_nca_pred))

                loss_disc = (loss_disc_real + loss_disc_fake_gen + loss_disc_fake_nca) / 3
                loss_disc.backward()
                opt_disc.step()

                # --- Train Generator and NCA ---
                generator.zero_grad()
                nca_model.zero_grad()

                # Re-generate images and get discriminator feedback
                fake_imgs_gen, w = generator(noise, return_w=True)
                disc_fake_gen_pred = discriminator(fake_imgs_gen)
                loss_gen_adv = F.binary_cross_entropy_with_logits(disc_fake_gen_pred, torch.ones_like(disc_fake_gen_pred))

                # Generate NCA images and get discriminator feedback
                seed = nca_model.get_seed(batch_size=real_imgs.shape[0], size=IMG_SIZE, device=DEVICE)
                nca_steps = random.randint(NCA_STEPS_MIN, NCA_STEPS_MAX)
                nca_output_grid = nca_model(seed, w, steps=nca_steps)
                fake_imgs_nca = nca_model.to_rgba(nca_output_grid)[:, :3, :, :] # Get RGB
                disc_fake_nca_pred = discriminator(fake_imgs_nca)
                loss_nca_adv = F.binary_cross_entropy_with_logits(disc_fake_nca_pred, torch.ones_like(disc_fake_nca_pred))
                
                # Matching loss: NCA tries to match the Generator's output
                loss_nca_match = F.mse_loss(fake_imgs_nca, fake_imgs_gen.detach())

                # Total loss for Generator and NCA
                loss_g_nca = loss_gen_adv + loss_nca_adv + (10.0 * loss_nca_match)
                loss_g_nca.backward()
                
                opt_gen.step()
                opt_nca.step()

                # Update status for UI
                if batch_idx % 10 == 0:
                    status_text = (
                        f"Epoch {epoch+1}/{EPOCHS}, "
                        f"Loss D: {loss_disc.item():.3f}, "
                        f"Loss G: {loss_gen_adv.item():.3f}, "
                        f"Loss NCA: {loss_nca_adv.item():.3f}, "
                        f"Loss Match: {loss_nca_match.item():.3f}"
                    )
                    progress_bar.set_postfix_str(status_text)
                    
                    with torch.no_grad():
                        sample_gen, sample_w = generator(fixed_noise, return_w=True)
                        seed = nca_model.get_seed(batch_size=fixed_noise.shape[0], size=IMG_SIZE, device=DEVICE)
                        sample_nca_grid = nca_model(seed, sample_w, steps=NCA_STEPS_MAX)
                        sample_nca = nca_model.to_rgba(sample_nca_grid)[:, :3, :, :]

                    # Pass images as a list of tensors
                    update_status(status_text, images=[sample_gen, sample_nca])

        update_status("Training finished.")
    except Exception as e:
        error_msg = f"Error in training loop: {str(e)}"
        print(error_msg)
        update_status(error_msg, error=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the NCA-StyleGAN application.")
    parser.add_argument('--run-training', action='store_true', help='Run the training loop.')
    args = parser.parse_args()

    if args.run_training:
        print("Starting training worker...")
        training_loop()
    else:
        print("This script is now only for training. Use --run-training to start.") 