import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

class EnhancedNCA(nn.Module):
    def __init__(self, channel_n=16, hidden_n=128, w_dim=128, activation='gelu'):
        super().__init__()
        self.channel_n = channel_n
        self.hidden_n = hidden_n
        self.w_dim = w_dim
        self.activation = activation
        
        # Perception network
        self.perception_net = nn.Sequential(
            nn.Conv2d(channel_n * 5, hidden_n, 1),  # Changed from 3 to 5 for correct stacking
            nn.ReLU(),
            nn.Conv2d(hidden_n, hidden_n, 1),
            nn.ReLU()
        )
        
        # Update network with selected activation
        self._setup_update_net()
        
    def _setup_update_net(self):
        if self.activation == 'gelu':
            self.update_net = nn.Sequential(
                nn.Conv2d(self.hidden_n + self.w_dim, self.hidden_n, 1),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_n),
                nn.Conv2d(self.hidden_n, self.channel_n, 1),
                nn.Tanh()
            )
        elif self.activation == 'leaky_relu':
            self.update_net = nn.Sequential(
                nn.Conv2d(self.hidden_n + self.w_dim, self.hidden_n, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(self.hidden_n),
                nn.Conv2d(self.hidden_n, self.channel_n, 1),
                nn.Tanh()
            )
        elif self.activation == 'relu':
            self.update_net = nn.Sequential(
                nn.Conv2d(self.hidden_n + self.w_dim, self.hidden_n, 1),
                nn.ReLU(),
                nn.BatchNorm2d(self.hidden_n),
                nn.Conv2d(self.hidden_n, self.channel_n, 1),
                nn.Tanh()
            )
    
    def get_seed(self, batch_size, size, device):
        x = torch.zeros(batch_size, self.channel_n, size, size, device=device)
        x[:, 3:, size//2-1:size//2+1, size//2-1:size//2+1] = 1.0
        return x
    
    def perceive(self, x):
        # Stack channels: normal, shifted up/down, shifted left/right
        y = torch.cat([
            x,  # Original
            torch.roll(x, shifts=1, dims=2),  # Up
            torch.roll(x, shifts=-1, dims=2),  # Down
            torch.roll(x, shifts=1, dims=3),  # Left
            torch.roll(x, shifts=-1, dims=3)  # Right
        ], dim=1)
        return self.perception_net(y)
    
    def forward(self, x, w, steps):
        batch_size = x.shape[0]
        w = w.view(batch_size, self.w_dim, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        for _ in range(steps):
            y = self.perceive(x)
            y = torch.cat([y, w], dim=1)
            dx = self.update_net(y)
            x = x + dx
            
        return x

def train_step(model, optimizer, x, w, target, steps=50):
    optimizer.zero_grad()
    output = model(x, w, steps)
    loss = torch.mean((output - target) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item()

def run_comparison(batch_size=4, img_size=64, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models with different activations
    models = {
        'gelu': EnhancedNCA(activation='gelu').to(device),
        'leaky_relu': EnhancedNCA(activation='leaky_relu').to(device),
        'relu': EnhancedNCA(activation='relu').to(device)
    }
    
    optimizers = {
        name: optim.Adam(model.parameters(), lr=1e-4)
        for name, model in models.items()
    }
    
    # Training history
    history = {name: [] for name in models.keys()}
    
    # Generate random target images
    target = torch.randn(batch_size, 16, img_size, img_size).to(device)
    target = torch.tanh(target)
    
    print("\nStarting training comparison...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_losses = {name: 0.0 for name in models.keys()}
        
        for name, model in models.items():
            x = model.get_seed(batch_size, img_size, device)
            w = torch.randn(batch_size, 128).to(device)
            
            loss = train_step(model, optimizers[name], x, w, target)
            history[name].append(loss)
            epoch_losses[name] = loss
            
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}:")
            for name, loss in epoch_losses.items():
                print(f"{name.upper():>10} Loss: {loss:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    # Calculate final statistics
    final_stats = {}
    for name, losses in history.items():
        final_stats[name] = {
            'final_loss': losses[-1],
            'mean_loss': sum(losses) / len(losses),
            'min_loss': min(losses),
            'max_loss': max(losses),
            'std_loss': torch.tensor(losses).std().item()
        }
    
    # Print final statistics
    print("\nFinal Statistics:")
    for name, stats in final_stats.items():
        print(f"\n{name.upper()}:")
        print(f"  Final Loss: {stats['final_loss']:.4f}")
        print(f"  Mean Loss:  {stats['mean_loss']:.4f}")
        print(f"  Min Loss:   {stats['min_loss']:.4f}")
        print(f"  Max Loss:   {stats['max_loss']:.4f}")
        print(f"  Std Dev:    {stats['std_loss']:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for name, losses in history.items():
        plt.plot(losses, label=f"{name} (final={losses[-1]:.4f})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Activation Function Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('activation_comparison.png')
    plt.close()

if __name__ == "__main__":
    run_comparison(epochs=200)  # Run for 200 epochs for better convergence 