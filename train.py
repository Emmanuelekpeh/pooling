import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import os
import numpy as np
import torch.nn.functional as F

from src.data_loader.loader import ImageDataset
from src.models.nca import NCAModel
from src.models.stylegan import Generator, Discriminator

# --- Device Configuration ---
def get_device():
    """Checks for CUDA, DirectML, and falls back to CPU."""
    if torch.cuda.is_available():
        print("Using CUDA GPU.")
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            print("Using DirectML GPU.")
            return torch_directml.device()
    except (ImportError, AttributeError):
        pass
    print("No GPU found, using CPU.")
    return torch.device("cpu")

DEVICE = get_device()
# --- End Device Configuration ---

# --- Configuration ---
DATA_DIR = "data/ukiyo-e"
IMG_SIZE = 128
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 100
Z_DIM = 128
W_DIM = 128
NCA_CHANNELS = 16

def train_nca(epochs=EPOCHS):
    print("--- Training NCA Model ---")
    
    # Setup
    dataset = ImageDataset(DATA_DIR, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = NCAModel(channel_n=NCA_CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    os.makedirs("samples/nca", exist_ok=True)

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for i, real_imgs in enumerate(pbar):
            real_imgs = real_imgs.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Pad real images to match NCA channel count
            target_rgba = F.pad(real_imgs, (0, 0, 0, 0, 1, 0), "constant", 1.0) # Add alpha channel
            target = F.pad(target_rgba, (0, 0, 0, 0, 0, NCA_CHANNELS - 4), "constant", 0.0)

            seed = model.get_seed(batch_size=real_imgs.shape[0], size=IMG_SIZE, device=DEVICE)
            
            # Grow the image
            generated_grid = model(seed, steps=np.random.randint(64, 96))
            generated_rgba = model.to_rgba(generated_grid)
            
            # Loss: Mean Squared Error between generated and target images
            loss = F.mse_loss(generated_rgba, target[:, :4, :, :])
            
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

            if i % 10 == 0:
                 # Save a sample of the generated images
                sample_grid = model.to_rgba(model(seed, steps=96))
                torchvision.utils.save_image(sample_grid, f"samples/nca/epoch_{epoch+1}_batch_{i}.png", normalize=True)

    print("NCA Training Finished.")


def train_stylegan(epochs=EPOCHS):
    print("--- Training StyleGAN Model ---")

    # Setup
    dataset = ImageDataset(DATA_DIR, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    gen = Generator(Z_DIM, W_DIM, img_size=IMG_SIZE).to(DEVICE)
    disc = Discriminator(img_size=IMG_SIZE).to(DEVICE)
    
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.99))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.0, 0.99))
    
    os.makedirs("samples/stylegan", exist_ok=True)
    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM).to(DEVICE)

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for i, real in enumerate(pbar):
            real = real.to(DEVICE)
            
            # Train Discriminator
            noise = torch.randn(real.shape[0], Z_DIM).to(DEVICE)
            fake = gen(noise)
            
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_real = F.relu(1.0 - disc_real).mean()
            loss_disc_fake = F.relu(1.0 + disc_fake).mean()
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            
            # Train Generator
            gen_fake = disc(fake).reshape(-1)
            loss_gen = -gen_fake.mean()
            
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss D: {loss_disc:.4f} | Loss G: {loss_gen:.4f}")

            if i % 10 == 0:
                with torch.no_grad():
                    fake_samples = gen(fixed_noise)
                    torchvision.utils.save_image(fake_samples, f"samples/stylegan/epoch_{epoch+1}_batch_{i}.png", normalize=True)

    print("StyleGAN Training Finished.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train NCA or StyleGAN models.")
    parser.add_argument('model', type=str, choices=['nca', 'stylegan'], help='Which model to train.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train for.')
    args = parser.parse_args()
    
    if args.model == 'nca':
        train_nca(epochs=args.epochs)
    elif args.model == 'stylegan':
        train_stylegan(epochs=args.epochs) 