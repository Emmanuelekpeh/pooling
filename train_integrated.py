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

# --- Project Configuration ---
DATA_DIR = "data/ukiyo-e"
IMG_SIZE = 64
BATCH_SIZE = 4 # Reduced batch size to lower memory usage
LR = 1e-4
EPOCHS = 500 # Increased epochs for longer runs
Z_DIM = 128
W_DIM = 128
NCA_CHANNELS = 16
NCA_STEPS_MIN, NCA_STEPS_MAX = 48, 64
SAMPLES_DIR = "samples/integrated"

# --- UI and Server Configuration ---
# We embed the HTML in the script for portability
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integrated Model Training</title>
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
    <h1>Ukiyo-e Project: Integrated Training Visualizer</h1>
    <p id="status">Connecting...</p>
    <div id="container">
        <div class="generation-box">
            <h2>StyleGAN Output</h2>
            <img id="stylegan-img" src="" alt="Waiting for StyleGAN generation...">
            <p id="stylegan-filename">No image generated yet.</p>
        </div>
        <div class="generation-box">
            <h2>NCA Output</h2>
            <img id="nca-img" src="" alt="Waiting for NCA generation...">
            <p id="nca-filename">No image generated yet.</p>
        </div>
    </div>
    <script>
        function updateImages() {
            fetch('/latest_images')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.status || 'Waiting for status...';
                    if (data.stylegan_image) {
                        const styleganImg = document.getElementById('stylegan-img');
                        const styleganFile = document.getElementById('stylegan-filename');
                        const newSrc = `/samples/${data.stylegan_image}?t=${new Date().getTime()}`;
                        if (styleganImg.src !== newSrc) {
                            styleganImg.src = newSrc;
                            styleganFile.textContent = data.stylegan_image;
                        }
                    }
                    if (data.nca_image) {
                        const ncaImg = document.getElementById('nca-img');
                        const ncaFile = document.getElementById('nca-filename');
                        const newSrc = `/samples/${data.nca_image}?t=${new Date().getTime()}`;
                        if (ncaImg.src !== newSrc) {
                            ncaImg.src = newSrc;
                            ncaFile.textContent = data.nca_image;
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching images:', error);
                    document.getElementById('status').textContent = 'Error connecting to server. Is the training running?';
                });
        }
        updateImages();
        setInterval(updateImages, 5000);
    </script>
</body>
</html>
"""

# --- Global state for communication between threads ---
training_status = "Initializing..."
latest_images_lock = threading.Lock()
latest_image_paths = {'stylegan_image': None, 'nca_image': None}


# --- Training Function (to be run in a background thread) ---
def run_training():
    global training_status, latest_image_paths
    
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
                        
                        with latest_images_lock:
                            latest_image_paths['stylegan_image'] = stylegan_fname
                            latest_image_paths['nca_image'] = nca_fname

        training_status = "Training Finished."

    except Exception as e:
        training_status = f"An error occurred: {e}"
        print(f"Error during training: {e}")

# --- Flask App ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/latest_images')
def get_latest_images():
    with latest_images_lock:
        response = latest_image_paths.copy()
    response['status'] = training_status
    return jsonify(response)

@app.route('/samples/<path:filename>')
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)

# --- Start Background Training ---
# This code runs once when the module is loaded. Gunicorn's --preload
# flag ensures this happens in a single master process before workers are forked.
print("Starting training thread...")
train_thread = threading.Thread(target=run_training, daemon=True)
train_thread.start()
print("Flask server is ready to be served by Gunicorn.") 