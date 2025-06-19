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
        h1 { color: #ffffff; font-weight: 300; }
        #container { display: flex; justify-content: center; align-items: flex-start; gap: 30px; margin-top: 30px; flex-wrap: wrap; }
        .generation-box { background-color: #1e1e1e; border-radius: 8px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); width: 400px; }
        h2 { margin-top: 0; color: #bb86fc; font-weight: 400; }
        img { max-width: 100%; border-radius: 4px; background-color: #2a2a2a; min-height: 256px; }
        p { font-size: 0.9em; color: #a0a0a0; word-wrap: break-word; }
        #status { margin-top: 20px; font-size: 1.1em; }
    </style>
</head>
<body>
    <h1>NCA-StyleGAN Competitive Training</h1>
    <div id="status">Server is up. Waiting to start training...</div>
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
        document.addEventListener('DOMContentLoaded', function() {
            // Start the training process once the page is loaded
            fetch('/start-training', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Training start response:', data.message);
                    // Now that training is started, begin polling for status and images
                    pollStatus();
                })
                .catch(error => {
                    console.error('Error starting training:', error);
                    document.getElementById('status').textContent = 'Error: Could not start training process.';
                });
        });

        function pollStatus() {
            setInterval(() => {
                // Fetch status
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').textContent = data.status;
                    });

                // Fetch images
                fetch('/latest-images')
                    .then(response => response.json())
                    .then(data => {
                        if (data.stylegan_image) {
                            document.getElementById('stylegan-image').src = 'data:image/png;base64,' + data.stylegan_image;
                        }
                        if (data.nca_image) {
                            document.getElementById('nca-image').src = 'data:image/png;base64,' + data.nca_image;
                        }
                    });
            }, 3000); // Poll every 3 seconds
        }
    </script>
</body>
</html>
"""

# --- Global State ---
training_status = "Idle"
latest_stylegan_img_b64 = None
latest_nca_img_b64 = None
TRAINING_STARTED = False
TRAINING_LOCK = threading.Lock()
# --- End Global State ---

# --- Training Function (to be run in a background thread) ---
def run_training():
    global training_status, latest_stylegan_img_b64, latest_nca_img_b64
    
    try:
        training_status = "Initializing models..."
        gen = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
        nca = IntegratedNCA(w_dim=W_DIM).to(DEVICE)
        disc = Discriminator(img_size=IMG_SIZE).to(DEVICE)

        gen_nca_params = list(gen.parameters()) + list(nca.parameters())
        opt_gen_nca = optim.Adam(gen_nca_params, lr=LR, betas=(0.0, 0.99))
        opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.0, 0.99))
        
        training_status = "Loading dataset..."
        # Note: num_workers > 0 can cause issues in some container environments. Setting to 0.
        # pin_memory is also set to False as we are on CPU.
        dataset = ImageDataset(DATA_DIR, img_size=IMG_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

        fixed_noise = torch.randn(min(BATCH_SIZE, 8), Z_DIM).to(DEVICE)
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        
        for epoch in range(EPOCHS):
            pbar = tqdm(dataloader)
            for i, real in enumerate(pbar):
                real = real.to(DEVICE)
                training_status = f"Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(dataloader)}"
                
                # --- Train Generator and NCA ---
                opt_gen_nca.zero_grad()

                # A. Generate StyleGAN image and the conditioning vector 'w'
                z = torch.randn(real.shape[0], Z_DIM).to(DEVICE)
                img_stylegan, w = gen(z, return_w=True)

                # B. NCA grows an image conditioned on 'w', trying to match the StyleGAN image
                seed = nca.get_seed(batch_size=real.shape[0], size=IMG_SIZE, device=DEVICE)
                nca_steps = torch.randint(NCA_STEPS_MIN, NCA_STEPS_MAX, (1,)).item()
                nca_grid = nca(seed, w.detach(), steps=nca_steps) # Detach w for NCA loss
                img_nca = nca.to_rgba(nca_grid)[:, :3, :, :] # Use RGB part for loss

                # C. Calculate Losses for the Generator/NCA network
                # C.1. NCA Loss: How well the NCA reproduced the StyleGAN image
                loss_nca = F.mse_loss(img_nca, img_stylegan.detach())

                # C.2. GAN Loss: How well the StyleGAN faked the discriminator
                disc_fake_stylegan = disc(img_stylegan).reshape(-1)
                loss_gan = -torch.mean(disc_fake_stylegan)
                
                # Total loss is a combination of both
                loss_gen_nca = loss_gan + loss_nca
                loss_gen_nca.backward()
                opt_gen_nca.step()

                # --- Train Discriminator ---
                opt_disc.zero_grad()
                
                # D.1. On real images
                disc_real = disc(real).reshape(-1)
                loss_disc_real = F.relu(1.0 - disc_real).mean()
                
                # D.2. On StyleGAN fakes
                disc_fake_stylegan = disc(img_stylegan.detach()).reshape(-1)
                loss_disc_fake_stylegan = F.relu(1.0 + disc_fake_stylegan).mean()
                
                # D.3. On NCA fakes
                disc_fake_nca = disc(img_nca.detach()).reshape(-1)
                loss_disc_fake_nca = F.relu(1.0 + disc_fake_nca).mean()
                
                # Combine discriminator losses
                loss_disc = (loss_disc_real + loss_disc_fake_stylegan + loss_disc_fake_nca) / 3
                loss_disc.backward()
                opt_disc.step()

                pbar.set_description(f"E:{epoch+1} | L_D:{loss_disc:.3f} | L_G/NCA:{loss_gen_nca:.3f} (G:{loss_gan:.3f}, N:{loss_nca:.3f})")

                if i % 50 == 0:
                    with torch.no_grad():
                        # Generate a batch of images with fixed noise to see progress
                        sample_stylegan, w_sample = gen(fixed_noise, return_w=True)
                        seed_sample = nca.get_seed(batch_size=fixed_noise.shape[0], size=IMG_SIZE, device=DEVICE)
                        nca_grid_sample = nca(seed_sample, w_sample, steps=NCA_STEPS_MAX)
                        sample_nca = nca.to_rgba(nca_grid_sample)[:,:3,:,:]
                        
                        ts = int(time.time())
                        stylegan_fname = f"epoch_{epoch+1}_ts_{ts}_stylegan.png"
                        nca_fname = f"epoch_{epoch+1}_ts_{ts}_nca.png"

                        torchvision.utils.save_image(sample_stylegan, os.path.join(SAMPLES_DIR, stylegan_fname), normalize=True)
                        torchvision.utils.save_image(sample_nca, os.path.join(SAMPLES_DIR, nca_fname), normalize=True)
                        
                        latest_stylegan_img_b64 = base64.b64encode(open(os.path.join(SAMPLES_DIR, stylegan_fname), 'rb').read()).decode('utf-8')
                        latest_nca_img_b64 = base64.b64encode(open(os.path.join(SAMPLES_DIR, nca_fname), 'rb').read()).decode('utf-8')

        training_status = "Training Finished."

    except Exception as e:
        training_status = f"An error occurred: {e}"
        print(f"Error during training: {e}")

# --- Flask App ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start-training', methods=['POST'])
def start_training():
    global TRAINING_STARTED
    with TRAINING_LOCK:
        if not TRAINING_STARTED:
            TRAINING_STARTED = True
            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()
            return jsonify({"message": "Training started successfully."}), 200
        else:
            return jsonify({"message": "Training has already been started."}), 200

@app.route('/status')
def get_status():
    return jsonify({"status": training_status})

@app.route('/latest-images')
def get_latest_images():
    return jsonify({"stylegan_image": latest_stylegan_img_b64, "nca_image": latest_nca_img_b64})

@app.route('/samples/<path:filename>')
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)

print("Flask server is ready to be served by Gunicorn.") 