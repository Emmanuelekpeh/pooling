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
Z_DIM = 64  # Reduced for memory
W_DIM = 64  # Reduced for memory
IMG_SIZE = 64
NCA_CHANNELS = 8  # Reduced for memory
BATCH_SIZE = 2  # Reduced batch size for stability
LR = 1e-4
EPOCHS = 100
NCA_STEPS_MIN = 64  # Increased from 32
NCA_STEPS_MAX = 96  # Increased from 48
DEVICE = "cpu" # Force CPU
DATA_DIR = "./data/ukiyo-e"
SAMPLES_DIR = "./samples"
# Use environment variable for checkpoint directory with fallback to relative path
# This ensures it works with the Fly.io mounted volume at /app/checkpoints
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints" if os.path.exists("/app") else "./checkpoints")

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
        self.const = nn.Parameter(torch.randn(1, 128, 4, 4)) # Further reduced
        self.up_block1 = nn.Upsample(scale_factor=2)
        self.gen_block1 = GeneratorBlock(128, 64) # Further reduced
        self.up_block2 = nn.Upsample(scale_factor=2)
        self.gen_block2 = GeneratorBlock(64, 32) # Further reduced
        self.up_block3 = nn.Upsample(scale_factor=2)
        self.gen_block3 = GeneratorBlock(32, 16) # Further reduced
        self.up_block4 = nn.Upsample(scale_factor=2)
        self.gen_block4 = GeneratorBlock(16, 8) # Additional layer for 64x64
        self.to_rgb = nn.Conv2d(8, 3, kernel_size=1)

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
            nn.Linear(hidden_n, channel_n),  # Add bias back
        )
        # Initialize the last layer with small weights
        with torch.no_grad():
            self.update_net[-1].weight.data *= 0.1
        
        # Register sobel filters as buffers so they move with the model
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_seed(self, batch_size, size, device):
        seed = torch.zeros(batch_size, self.channel_n, size, size, device=device)
        # Create a larger initial seed pattern (5x5 area) instead of just a single pixel
        center = size // 2
        radius = 2  # Create a 5x5 area (center ± 2)
        
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                # Set RGB values with some variation
                seed[:, 0, center+i, center+j] = 0.5 + torch.rand(batch_size, device=device) * 0.2  # R
                seed[:, 1, center+i, center+j] = 0.3 + torch.rand(batch_size, device=device) * 0.2  # G
                seed[:, 2, center+i, center+j] = 0.7 + torch.rand(batch_size, device=device) * 0.2  # B
                seed[:, 3, center+i, center+j] = 1.0  # Alpha (alive)
        
        # Add some noise to hidden channels
        if self.channel_n > 4:
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    seed[:, 4:, center+i, center+j] = torch.randn(batch_size, self.channel_n - 4, device=device) * 0.1
        return seed

    def to_rgba(self, x):
        # Clamp RGB values to [-1, 1] range and ensure alpha is positive
        rgba = x[:, :4, :, :].clone()
        rgba[:, :3, :, :] = torch.tanh(rgba[:, :3, :, :])  # RGB in [-1, 1]
        rgba[:, 3:4, :, :] = torch.sigmoid(rgba[:, 3:4, :, :])  # Alpha in [0, 1]
        return rgba

    def perceive(self, x):
        grad_x = F.conv2d(x, self.sobel_x.repeat(self.channel_n, 1, 1, 1), padding=1, groups=self.channel_n)
        grad_y = F.conv2d(x, self.sobel_y.repeat(self.channel_n, 1, 1, 1), padding=1, groups=self.channel_n)
        return torch.cat((x, grad_x, grad_y), 1)

    def forward(self, x, w, steps):
        for step in range(steps):
            # Get living mask - cells are alive if alpha > 0.1
            alive_mask = (x[:, 3:4, :, :] > 0.1).float()
            
            # Perception
            perceived = self.perceive(x)
            perceived = perceived.permute(0, 2, 3, 1).reshape(-1, self.channel_n * 3)
            
            # Expand w to match spatial dimensions
            w_expanded = w.unsqueeze(1).unsqueeze(1).repeat(1, x.shape[2], x.shape[3], 1)
            w_reshaped = w_expanded.reshape(-1, self.w_dim)
            
            # Combine perception with style vector
            update_input = torch.cat([perceived, w_reshaped], dim=1)
            ds = self.update_net(update_input)
            
            # Reshape back to spatial format
            ds = ds.reshape(x.shape[0], x.shape[2], x.shape[3], self.channel_n).permute(0, 3, 1, 2)
            
            # More aggressive update - increase probability of updates
            stochastic_mask = (torch.rand_like(alive_mask) < 0.8).float()  # Increased from 0.5 to 0.8
            update_mask = alive_mask * stochastic_mask
            
            # Apply update with stronger effect
            x = x + ds * update_mask * 0.2  # Increased from 0.1 to 0.2
            
            # More permissive life conditions - use max pooling with larger kernel
            neighbor_life = F.max_pool2d(alive_mask, kernel_size=5, stride=1, padding=2)  # Increased kernel from 3 to 5
            life_mask = (neighbor_life > 0.05).float()  # More permissive threshold (0.1 to 0.05)
            
            # Keep cells alive and apply life mask
            x = x * life_mask
            
        return x

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), # 64 -> 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1), # 32 -> 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), # 16 -> 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), # 8 -> 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 0), # 4 -> 1 (final layer)
        )
    def forward(self, img):
        return self.model(img).view(-1)

# --- Cross-Evaluation Networks ---
class CrossEvaluator(nn.Module):
    """Network that evaluates the quality of images from the other model"""
    def __init__(self, img_size):
        super().__init__()
        self.evaluator = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64 -> 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), # 32 -> 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), # 16 -> 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), # 8 -> 4
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Quality score 0-1
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.evaluator(img).squeeze()

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

# --- Checkpoint Management ---
def save_checkpoint(epoch, models, optimizers, losses, scores, checkpoint_dir):
    """Save training checkpoint"""
    try:
        print(f"Attempting to save checkpoint to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True, mode=0o777)  # Ensure directory has write permissions
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': models['generator'].state_dict(),
            'nca_state_dict': models['nca'].state_dict(),
            'discriminator_state_dict': models['discriminator'].state_dict(),
            'gen_evaluator_state_dict': models['gen_evaluator'].state_dict(),
            'nca_evaluator_state_dict': models['nca_evaluator'].state_dict(),
            'opt_gen_state_dict': optimizers['gen'].state_dict(),
            'opt_nca_state_dict': optimizers['nca'].state_dict(),
            'opt_disc_state_dict': optimizers['disc'].state_dict(),
            'opt_gen_eval_state_dict': optimizers['gen_eval'].state_dict(),
            'opt_nca_eval_state_dict': optimizers['nca_eval'].state_dict(),
            'losses': losses,
            'scores': scores
        }
        
        # Save epoch-specific checkpoint
        epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)
        print(f"Saved checkpoint to {epoch_path}")
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")
        
        # Verify files exist
        if os.path.exists(epoch_path) and os.path.exists(latest_path):
            print(f"Checkpoint saved at epoch {epoch} - Verified files exist")
            print(f"Epoch checkpoint size: {os.path.getsize(epoch_path)} bytes")
            print(f"Latest checkpoint size: {os.path.getsize(latest_path)} bytes")
        else:
            print(f"Warning: Checkpoint files not found after saving")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        # Try to create a simple test file to check permissions
        try:
            test_file = os.path.join(checkpoint_dir, "test_write.txt")
            with open(test_file, "w") as f:
                f.write("Test write access")
            print(f"Successfully created test file at {test_file}")
            os.remove(test_file)
        except Exception as test_e:
            print(f"Error creating test file: {str(test_e)}")

def load_checkpoint(checkpoint_path, models, optimizers):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found, starting from scratch")
        return 0, [], []
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    models['generator'].load_state_dict(checkpoint['generator_state_dict'])
    models['nca'].load_state_dict(checkpoint['nca_state_dict'])
    models['discriminator'].load_state_dict(checkpoint['discriminator_state_dict'])
    models['gen_evaluator'].load_state_dict(checkpoint['gen_evaluator_state_dict'])
    models['nca_evaluator'].load_state_dict(checkpoint['nca_evaluator_state_dict'])
    
    optimizers['gen'].load_state_dict(checkpoint['opt_gen_state_dict'])
    optimizers['nca'].load_state_dict(checkpoint['opt_nca_state_dict'])
    optimizers['disc'].load_state_dict(checkpoint['opt_disc_state_dict'])
    optimizers['gen_eval'].load_state_dict(checkpoint['opt_gen_eval_state_dict'])
    optimizers['nca_eval'].load_state_dict(checkpoint['opt_nca_eval_state_dict'])
    
    epoch = checkpoint['epoch']
    losses = checkpoint.get('losses', [])
    scores = checkpoint.get('scores', [])
    
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, losses, scores

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NCA vs StyleGAN Training</title>
    <style>
        body { font-family: sans-serif; background-color: #282c34; color: #abb2bf; text-align: center; }
        .container { display: flex; justify-content: center; align-items: flex-start; gap: 40px; margin-top: 20px; flex-wrap: wrap; }
        @media (max-width: 2500px) { .container { flex-direction: column; align-items: center; } }
        .column { display: flex; flex-direction: column; align-items: center; }
        h1, h2 { color: #61afef; }
        img { border: 2px solid #61afef; width: 800px; height: 800px; image-rendering: pixelated; object-fit: contain; max-width: none !important; max-height: none !important; }
        #status { margin-top: 20px; font-size: 1.2em; min-height: 50px; padding: 10px; border-radius: 5px; background-color: #3b4048; }
        .error { color: #e06c75; border: 1px solid #e06c75; }
        .scores { margin-top: 10px; font-size: 0.9em; color: #98c379; }
    </style>
</head>
<body>
    <h1>NCA vs StyleGAN Training Status</h1>
    <div id="status">Connecting...</div>
    <div class="container">
        <div class="column">
            <h2>Real Target Images</h2>
            <img id="target-image" src="https://via.placeholder.com/400" alt="Target Images">
        </div>
        <div class="column">
            <h2>Generator Output</h2>
            <img id="generator-image" src="https://via.placeholder.com/400" alt="Generator Output">
            <div id="gen-scores" class="scores">Quality Score: --</div>
        </div>
        <div class="column">
            <h2>NCA Output</h2>
            <img id="nca-image" src="https://via.placeholder.com/400" alt="NCA Output">
            <div id="nca-scores" class="scores">Quality Score: --</div>
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
                            if (data.images[0]) document.getElementById('target-image').src = 'data:image/png;base64,' + data.images[0];
                            if (data.images[1]) document.getElementById('generator-image').src = 'data:image/png;base64,' + data.images[1];
                            if (data.images[2]) document.getElementById('nca-image').src = 'data:image/png;base64,' + data.images[2];
                        }
                        
                        if (data.scores) {
                            document.getElementById('gen-scores').textContent = `Quality Score: ${data.scores.gen_quality?.toFixed(3) || '--'}`;
                            document.getElementById('nca-scores').textContent = `Quality Score: ${data.scores.nca_quality?.toFixed(3) || '--'}`;
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
    """Converts the first tensor in a batch to a large base64 encoded PNG image."""
    # Ensure tensor is on CPU and detached from the computation graph
    tensor = tensor.detach().cpu()
    # Take only the first image from the batch and normalize it
    single_img = tensor[0]  # Shape: [3, 64, 64]
    # Normalize from [-1, 1] to [0, 1]
    single_img = (single_img + 1.0) / 2.0
    # Clamp and convert to uint8
    ndarr = single_img.mul(255).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    # Create PIL Image and resize to make it much larger
    im = Image.fromarray(ndarr)
    # Resize with nearest neighbor for crisp pixels
    im = im.resize((512, 512), Image.NEAREST)
    # Save to buffer
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def update_status(status_message, images=None, scores=None, error=False):
    """Writes the current status, images, and scores to a JSON file."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    # The 'images' argument is expected to be a list of tensors
    b64_images = [tensor_to_b64(img_tensor) for img_tensor in images] if images else []
    with open(STATUS_FILE, 'w') as f:
        json.dump({
            'status': status_message,
            'images': b64_images,
            'scores': scores or {},
            'error': error,
            'timestamp': time.time()
        }, f)

# --- Training Function (to be run by the 'worker' process) ---
def training_loop():
    """The main training loop with persistence and mutual evaluation."""
    try:
        update_status("Initializing models and data...")
        
        # Print checkpoint directory information
        print(f"Checkpoint directory: {CHECKPOINT_DIR}")
        if os.path.exists(CHECKPOINT_DIR):
            print(f"Checkpoint directory exists. Contents: {os.listdir(CHECKPOINT_DIR)}")
            print(f"Permissions: {oct(os.stat(CHECKPOINT_DIR).st_mode & 0o777)}")
        else:
            print(f"Checkpoint directory does not exist. Creating it...")
            try:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True, mode=0o777)
                print(f"Created checkpoint directory. Permissions: {oct(os.stat(CHECKPOINT_DIR).st_mode & 0o777)}")
            except Exception as e:
                print(f"Error creating checkpoint directory: {str(e)}")
        
        # --- Create separate models and optimizers ---
        generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
        nca_model = IntegratedNCA(channel_n=NCA_CHANNELS, w_dim=W_DIM).to(DEVICE)
        discriminator = Discriminator(IMG_SIZE).to(DEVICE)
        
        # Cross-evaluation networks
        gen_evaluator = CrossEvaluator(IMG_SIZE).to(DEVICE)  # Evaluates NCA outputs
        nca_evaluator = CrossEvaluator(IMG_SIZE).to(DEVICE)  # Evaluates Generator outputs

        models = {
            'generator': generator,
            'nca': nca_model,
            'discriminator': discriminator,
            'gen_evaluator': gen_evaluator,
            'nca_evaluator': nca_evaluator
        }

        # Optimizers
        opt_gen = optim.Adam(generator.mapping_network.parameters(), lr=LR, betas=(0.5, 0.999))
        opt_nca = optim.Adam(nca_model.parameters(), lr=LR * 10, betas=(0.5, 0.999))
        opt_disc = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
        opt_gen_eval = optim.Adam(gen_evaluator.parameters(), lr=LR, betas=(0.5, 0.999))
        opt_nca_eval = optim.Adam(nca_evaluator.parameters(), lr=LR, betas=(0.5, 0.999))

        optimizers = {
            'gen': opt_gen,
            'nca': opt_nca,
            'disc': opt_disc,
            'gen_eval': opt_gen_eval,
            'nca_eval': opt_nca_eval
        }
        
        # Load checkpoint if exists
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
        start_epoch, loss_history, score_history = load_checkpoint(checkpoint_path, models, optimizers)
        
        dataset = ImageDataset(DATA_DIR, img_size=IMG_SIZE)
        print(f"Found {len(dataset)} images in dataset")
        if len(dataset) == 0:
            raise ValueError(f"No images found in {DATA_DIR}")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

        fixed_noise = torch.randn(BATCH_SIZE, Z_DIM).to(DEVICE)
        
        update_status(f"Resuming training from epoch {start_epoch + 1}...")
        
        print(f"Training configuration:")
        print(f"- IMG_SIZE: {IMG_SIZE}")
        print(f"- BATCH_SIZE: {BATCH_SIZE}")
        print(f"- DEVICE: {DEVICE}")
        print(f"- Dataset size: {len(dataset)}")
        print(f"- Starting from epoch: {start_epoch + 1}")
        
        for epoch in range(start_epoch, EPOCHS):
            epoch_scores = {'gen_quality': 0.0, 'nca_quality': 0.0, 'gen_eval_loss': 0.0, 'nca_eval_loss': 0.0}
            batch_count = 0
            
            progress_bar = tqdm(dataloader, desc=f"E:{epoch+1}", leave=False)
            for batch_idx, real_imgs in enumerate(progress_bar):
                real_imgs = real_imgs.to(DEVICE)
                batch_count += 1
                
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
                    nca_rgba = nca_model.to_rgba(nca_output_grid)
                    fake_imgs_nca = nca_rgba[:, :3, :, :] * 2.0 - 1.0  # Convert [0,1] to [-1,1] to match real images
                disc_fake_nca_pred = discriminator(fake_imgs_nca.detach())
                loss_disc_fake_nca = F.binary_cross_entropy_with_logits(disc_fake_nca_pred, torch.zeros_like(disc_fake_nca_pred))

                loss_disc = (loss_disc_real + loss_disc_fake_gen + loss_disc_fake_nca) / 3
                loss_disc.backward()
                opt_disc.step()

                # --- Train Cross-Evaluators ---
                gen_evaluator.zero_grad()
                nca_evaluator.zero_grad()
                
                # Train Generator's evaluator on NCA outputs (teach it to score NCA quality)
                nca_quality_pred = gen_evaluator(fake_imgs_nca.detach())
                # Use discriminator's opinion as ground truth for quality
                with torch.no_grad():
                    nca_quality_target = torch.sigmoid(disc_fake_nca_pred).detach()
                loss_gen_eval = F.mse_loss(nca_quality_pred, nca_quality_target)
                loss_gen_eval.backward()
                opt_gen_eval.step()
                
                # Train NCA's evaluator on Generator outputs
                gen_quality_pred = nca_evaluator(fake_imgs_gen.detach())
                with torch.no_grad():
                    gen_quality_target = torch.sigmoid(disc_fake_gen_pred).detach()
                loss_nca_eval = F.mse_loss(gen_quality_pred, gen_quality_target)
                loss_nca_eval.backward()
                opt_nca_eval.step()

                # --- Train Generator and NCA with Mutual Evaluation ---
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
                nca_rgba = nca_model.to_rgba(nca_output_grid)
                fake_imgs_nca = nca_rgba[:, :3, :, :] * 2.0 - 1.0  # Convert [0,1] to [-1,1] to match real images
                disc_fake_nca_pred = discriminator(fake_imgs_nca)
                loss_nca_adv = F.binary_cross_entropy_with_logits(disc_fake_nca_pred, torch.ones_like(disc_fake_nca_pred))
                
                # Mutual evaluation losses
                gen_scores_nca = nca_evaluator(fake_imgs_gen)  # NCA evaluates Generator
                nca_scores_gen = gen_evaluator(fake_imgs_nca)  # Generator evaluates NCA
                
                # Encourage high mutual evaluation scores
                loss_gen_mutual = -gen_scores_nca.mean()  # Generator wants NCA to score it highly
                loss_nca_mutual = -nca_scores_gen.mean()  # NCA wants Generator to score it highly
                
                # Matching loss: NCA tries to match the Generator's output
                loss_nca_match = F.mse_loss(fake_imgs_nca, fake_imgs_gen.detach())

                # Combined losses with mutual evaluation
                loss_gen_total = loss_gen_adv + 0.1 * loss_gen_mutual
                loss_nca_total = loss_nca_adv + 0.1 * loss_nca_mutual + 1.0 * loss_nca_match
                
                loss_total = loss_gen_total + loss_nca_total
                loss_total.backward()
                
                opt_gen.step()
                opt_nca.step()

                # Accumulate scores for this epoch
                with torch.no_grad():
                    epoch_scores['gen_quality'] += gen_scores_nca.mean().item()
                    epoch_scores['nca_quality'] += nca_scores_gen.mean().item()
                    epoch_scores['gen_eval_loss'] += loss_gen_eval.item()
                    epoch_scores['nca_eval_loss'] += loss_nca_eval.item()

                # Update status for UI
                if batch_idx % 10 == 0:
                    current_scores = {
                        'gen_quality': gen_scores_nca.mean().item(),
                        'nca_quality': nca_scores_gen.mean().item()
                    }
                    
                    status_text = (
                        f"Epoch {epoch+1}/{EPOCHS}, "
                        f"Loss D: {loss_disc.item():.3f}, "
                        f"Loss G: {loss_gen_adv.item():.3f}, "
                        f"Loss NCA: {loss_nca_adv.item():.3f}, "
                        f"Gen Quality: {current_scores['gen_quality']:.3f}, "
                        f"NCA Quality: {current_scores['nca_quality']:.3f}"
                    )
                    progress_bar.set_postfix_str(status_text)
                    
                    with torch.no_grad():
                        sample_gen, sample_w = generator(fixed_noise, return_w=True)
                        seed = nca_model.get_seed(batch_size=fixed_noise.shape[0], size=IMG_SIZE, device=DEVICE)
                        sample_nca_grid = nca_model(seed, sample_w, steps=NCA_STEPS_MAX)
                        sample_nca_rgba = nca_model.to_rgba(sample_nca_grid)
                        sample_nca = sample_nca_rgba[:, :3, :, :] * 2.0 - 1.0  # Convert [0,1] to [-1,1] for display

                    # Pass images as a list of tensors: [target, generator, nca]
                    update_status(status_text, images=[real_imgs, sample_gen, sample_nca], scores=current_scores)

            # Average scores for this epoch
            for key in epoch_scores:
                epoch_scores[key] /= batch_count
            
            score_history.append(epoch_scores)
            
            # Save checkpoint more frequently in early epochs, then every 5 epochs
            save_checkpoint_now = False
            if epoch < 5:  # First 5 epochs, save after each epoch
                save_checkpoint_now = True
            elif (epoch + 1) % 5 == 0:  # Then every 5 epochs
                save_checkpoint_now = True
                
            if save_checkpoint_now:
                save_checkpoint(epoch + 1, models, optimizers, loss_history, score_history, CHECKPOINT_DIR)

        update_status("Training finished.")
        save_checkpoint(EPOCHS, models, optimizers, loss_history, score_history, CHECKPOINT_DIR)
        
    except Exception as e:
        error_msg = f"Error in training loop: {str(e)}"
        print(error_msg)
        update_status(error_msg, error=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the NCA-StyleGAN application.")
    parser.add_argument('--run-training', action='store_true', help='Run the training loop.')
    parser.add_argument('--test-checkpoint', action='store_true', help='Test checkpoint functionality.')
    args = parser.parse_args()

    if args.test_checkpoint:
        print("Testing checkpoint functionality...")
        # Create a simple model to test checkpoint saving
        test_model = nn.Linear(10, 5)
        test_optimizer = optim.Adam(test_model.parameters(), lr=0.001)
        
        # Create a test checkpoint
        test_checkpoint = {
            'epoch': 0,
            'model_state_dict': test_model.state_dict(),
            'optimizer_state_dict': test_optimizer.state_dict(),
            'test_data': [1, 2, 3, 4, 5]
        }
        
        # Print checkpoint directory information
        print(f"Checkpoint directory: {CHECKPOINT_DIR}")
        if os.path.exists(CHECKPOINT_DIR):
            print(f"Checkpoint directory exists. Contents: {os.listdir(CHECKPOINT_DIR)}")
            print(f"Permissions: {oct(os.stat(CHECKPOINT_DIR).st_mode & 0o777)}")
        else:
            print(f"Checkpoint directory does not exist. Creating it...")
            try:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True, mode=0o777)
                print(f"Created checkpoint directory. Permissions: {oct(os.stat(CHECKPOINT_DIR).st_mode & 0o777)}")
            except Exception as e:
                print(f"Error creating checkpoint directory: {str(e)}")
        
        # Save test checkpoint
        test_path = os.path.join(CHECKPOINT_DIR, 'test_checkpoint.pt')
        try:
            torch.save(test_checkpoint, test_path)
            print(f"Successfully saved test checkpoint to {test_path}")
            
            # Verify file exists and load it
            if os.path.exists(test_path):
                print(f"Test checkpoint exists with size: {os.path.getsize(test_path)} bytes")
                loaded = torch.load(test_path)
                print(f"Successfully loaded test checkpoint with keys: {list(loaded.keys())}")
            else:
                print("Error: Test checkpoint file not found after saving")
        except Exception as e:
            print(f"Error during checkpoint test: {str(e)}")
            # Try to create a simple test file
            try:
                test_file = os.path.join(CHECKPOINT_DIR, "test_write.txt")
                with open(test_file, "w") as f:
                    f.write("Test write access")
                print(f"Successfully created test file at {test_file}")
            except Exception as test_e:
                print(f"Error creating test file: {str(test_e)}")
    elif args.run_training:
        print("Starting training worker...")
        training_loop()
    else:
        print("This script is now only for training. Use --run-training to start or --test-checkpoint to test checkpoint functionality.") 