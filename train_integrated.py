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
    </style>
</head>
<body>
    <h1>NCA-StyleGAN Competitive Training</h1>
    <div id="status">Connecting to server...</div>
    <div class="container">
        <div class="image-box">
            <h2>StyleGAN Output</h2>
            <img id="stylegan-image" src="" alt="StyleGAN Output">
        </div>
        <div class="image-box">
            <h2>NCA Output</h2>
            <img id="nca-image" src="" alt="NCA Output">
        </div>
    </div>

    <script>
        function pollStatus() {
            setInterval(() => {
                // Fetch status and images together
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').textContent = data.status;
                        if (data.stylegan_image) {
                            document.getElementById('stylegan-image').src = 'data:image/png;base64,' + data.stylegan_image;
                        }
                        if (data.nca_image) {
                            document.getElementById('nca-image').src = 'data:image/png;base64,' + data.nca_image;
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching status:', error);
                        document.getElementById('status').textContent = 'Error: Could not connect to the server.';
                    });
            }, 3000); // Poll every 3 seconds
        }
        // Start polling as soon as the page loads
        document.addEventListener('DOMContentLoaded', pollStatus);
    </script>
</body>
</html>
"""

# --- File-based state for inter-process communication ---
STATUS_FILE = os.path.join(SAMPLES_DIR, 'status.json')

def update_status(status_message, stylegan_b64=None, nca_b64=None):
    """Writes the current status to a JSON file."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump({
            'status': status_message,
            'stylegan_image': stylegan_b64,
            'nca_image': nca_b64
        }, f)

# --- Training Function (to be run by the 'worker' process) ---
def training_loop():
    """The main training loop."""
    try:
        update_status("Initializing models and data...")
        gen_nca = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
        disc = Discriminator(IMG_SIZE).to(DEVICE)

        gen_nca_params = list(gen_nca.mapping_network.parameters()) + list(gen_nca.nca.parameters())
        opt_gen_nca = optim.Adam(gen_nca_params, lr=LR, betas=(0.0, 0.99))
        opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.0, 0.99))

        update_status("Loading dataset...")
        dataset = ImageDataset(DATA_DIR, img_size=IMG_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

        fixed_noise = torch.randn(min(BATCH_SIZE, 8), Z_DIM).to(DEVICE)
        
        update_status("Starting training...")
        for epoch in range(EPOCHS):
            for batch_idx, real in enumerate(tqdm(dataloader, desc=f"E:{epoch+1}")):
                real = real.to(DEVICE)
                noise = torch.randn(real.size(0), Z_DIM).to(DEVICE)

                # Train Discriminator
                opt_disc.zero_grad()
                stylegan_img, nca_img = gen_nca(noise, steps=random.randint(NCA_STEPS_MIN, NCA_STEPS_MAX))
                
                disc_real = disc(real).view(-1)
                loss_disc_real = -torch.mean(torch.min(torch.zeros_like(disc_real), -1.0 + disc_real))
                
                disc_fake_stylegan = disc(stylegan_img.detach()).view(-1)
                loss_disc_fake_stylegan = -torch.mean(torch.min(torch.zeros_like(disc_fake_stylegan), -1.0 - disc_fake_stylegan))

                disc_fake_nca = disc(nca_img.detach()).view(-1)
                loss_disc_fake_nca = -torch.mean(torch.min(torch.zeros_like(disc_fake_nca), -1.0 - disc_fake_nca))
                
                loss_disc = (loss_disc_real + loss_disc_fake_stylegan + loss_disc_fake_nca) / 3
                loss_disc.backward()
                opt_disc.step()

                # Train Generator and NCA
                opt_gen_nca.zero_grad()
                output_fake_stylegan = disc(stylegan_img).view(-1)
                output_fake_nca = disc(nca_img).view(-1)
                loss_g_stylegan = -torch.mean(output_fake_stylegan)
                loss_g_nca = -torch.mean(output_fake_nca)
                
                match_loss = torch.mean(torch.abs(stylegan_img - nca_img))
                loss_gen = loss_g_stylegan + loss_g_nca + match_loss
                loss_gen.backward()
                opt_gen_nca.step()

                # Update status and generate sample images
                if batch_idx % 10 == 0:
                    status_msg = f"E:{epoch+1} | L_D:{loss_disc:.3f} | L_G/NCA:{loss_gen:.3f} (G_S:{loss_g_stylegan:.3f}, G_N:{loss_g_nca:.3f}, M:{match_loss:.3f})"
                    
                    with torch.no_grad():
                        sample_stylegan, sample_nca = gen_nca(fixed_noise, steps=NCA_STEPS_MAX)
                        
                        # Convert tensors to b64 strings
                        stylegan_b64 = tensor_to_b64(sample_stylegan)
                        nca_b64 = tensor_to_b64(sample_nca)
                        update_status(status_msg, stylegan_b64, nca_b64)

        update_status("Training Finished.")
    except Exception as e:
        print(f"Error in training loop: {e}")
        update_status(f"Error: {e}")

def tensor_to_b64(tensor):
    """Converts a tensor to a base64 encoded string."""
    # Denormalize and save to a bytes buffer
    img_grid = transforms.ToPILImage()(torch.clamp((tensor[0] + 1) / 2.0, 0, 1).cpu())
    buffered = io.BytesIO()
    img_grid.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Web Server (to be run by the 'web' process) ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def get_status():
    """Reads status from the JSON file and returns it."""
    if not os.path.exists(STATUS_FILE):
        return jsonify({
            "status": "Worker has not started yet. Please wait.",
            "stylegan_image": None,
            "nca_image": None
        })
    try:
        with open(STATUS_FILE, 'r') as f:
            return jsonify(json.load(f))
    except (IOError, json.JSONDecodeError) as e:
        return jsonify({
            "status": f"Error reading status file: {e}",
            "stylegan_image": None,
            "nca_image": None
        })

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