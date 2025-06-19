import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import os
import torch.nn.functional as F
from flask import Flask, render_template_string, jsonify, send_from_directory
import threading
import glob
import time
import io
import base64
import random
import resource # Import the resource module
import argparse # For command-line argument parsing
import json     # For status communication
from PIL import Image

from src.data_loader.loader import ImageDataset
from src.models.integrated import IntegratedGenerator, IntegratedNCA
from src.models.stylegan import Discriminator # We can reuse the original discriminator

# --- Device Configuration ---
def get_device():
    """Checks for CUDA and falls back to CPU."""
    # Standard CUDA check
    if torch.cuda.is_available():
        print("Using CUDA GPU.")
        return torch.device("cuda")
    
    print("No GPU found, using CPU.")
    return torch.device("cpu")

DEVICE = get_device()
# --- End Device Configuration ---

# --- Memory Limit Configuration ---
def set_memory_limit():
    """
    Reads the container's memory limit and sets it for the Python process.
    This helps in getting a MemoryError instead of a sudden OOM kill.
    """
    # This path is specific to cgroup v1, which is common in many container envs.
    # Fly.io uses cgroup v2, where the path is different. We'll check for both.
    cgroup_v1_path = '/sys/fs/cgroup/memory/memory.limit_in_bytes'
    cgroup_v2_path = '/sys/fs/cgroup/memory.max' # Path for cgroup v2

    mem_limit = -1
    try:
        if os.path.exists(cgroup_v1_path):
            with open(cgroup_v1_path) as f:
                limit_str = f.read().strip()
                # If the limit is ridiculously high, it means no limit.
                if int(limit_str) < 10**18: 
                    mem_limit = int(limit_str)
        elif os.path.exists(cgroup_v2_path):
            with open(cgroup_v2_path) as f:
                limit_str = f.read().strip()
                if limit_str != 'max':
                     mem_limit = int(limit_str)

        if mem_limit > 0:
            # Set the soft and hard limits for the address space
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            print(f"Container memory limit set to: {mem_limit / 1024 / 1024:.2f} MB")
        else:
            print("No container memory limit found or it's unlimited.")

    except (IOError, ValueError) as e:
        print(f"Could not set memory limit: {e}")

# Call this at the start of the application
set_memory_limit()
# --- End Memory Limit Configuration ---

# --- Project Configuration ---
DATA_DIR = "data/ukiyo-e"
IMG_SIZE = 64 # Drastically reduced image size to lower memory
BATCH_SIZE = 4 # Reduced batch size to lower memory usage
LR = 1e-4
EPOCHS = 500 # Increased epochs for longer runs
Z_DIM = 64 # Reduced model complexity
W_DIM = 64 # Reduced model complexity
NCA_CHANNELS = 12 # Reduced model complexity
NCA_STEPS_MIN, NCA_STEPS_MAX = 48, 64
SAMPLES_DIR = "samples/integrated"

# --- UI and Server Configuration ---
# We use render_template_string to keep everything in one file for deployment simplicity.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NCA-StyleGAN Training</title>
    <style>
        body { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; text-align: center; }
        h1, h2 { color: #ffffff; }
        .container { display: flex; justify-content: center; gap: 20px; margin-top: 20px; flex-wrap: wrap; }
        .image-box { background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; width: 300px; }
        img { max-width: 100%; height: auto; border-radius: 4px; margin-top: 10px; background-color: #2a2a2a; }
        #status { margin-top: 20px; font-size: 1.1em; background-color: #1e1e1e; padding: 10px; border-radius: 5px; display: inline-block; }
        #error { color: #ff6b6b; font-weight: bold; }
    </style>
</head>
<body>
    <h1>NCA-StyleGAN Competitive Training</h1>
    <div id="status">Connecting to server...</div>
    <div class="container">
        <div class="image-box">
            <h2>Generator Output</h2>
            <img id="generator-image" src="" alt="Generator Output">
        </div>
        <div class="image-box">
            <h2>NCA Output</h2>
            <img id="nca-image" src="" alt="NCA Output">
        </div>
    </div>

    <script>
        function pollStatus() {
            setInterval(() => {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        statusDiv.textContent = data.status;
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

# --- Web Server (to be run by the 'web' process) ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def get_status():
    """Reads the status from the JSON file."""
    if not os.path.exists(STATUS_FILE):
        return jsonify({'status': 'Worker has not started yet.', 'images': [], 'error': False})
    
    try:
        with open(STATUS_FILE, 'r') as f:
            return jsonify(json.load(f))
    except (IOError, json.JSONDecodeError):
        return jsonify({'status': 'Error reading status file.', 'images': [], 'error': True})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the NCA-StyleGAN application.")
    parser.add_argument('--run-training', action='store_true', help='Run the training loop.')
    args = parser.parse_args()

    if args.run_training:
        print("Starting training worker...")
        training_loop()
    else:
        # This case is for local testing of the web server, Gunicorn will run the 'app' object directly.
        print("Starting Flask web server for local testing...")
        # Initialize status file for web server
        update_status("Web server started. Waiting for worker to start training.")
        app.run(debug=True, port=5001) 