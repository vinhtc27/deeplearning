import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

eval_only = True
batch_size = 64
epochs = 50
latent_dim = 32
lr = 1e-3
model_path = "./output/ae/ae_best.pt"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 784 → 256 → 64 → latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder: latent_dim → 64 → 256 → 784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

model = Autoencoder(latent_dim=latent_dim).to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

criterion = nn.MSELoss()
if eval_only:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Set eval_only = False to train first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")
else:
    best_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                test_loss += loss.item() * images.size(0)
        test_loss /= len(test_loader.dataset)
        test_loss_history.append(test_loss)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f} Time: {epoch_time:.2f}s")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f"🔥 New best model saved! ({best_loss:.6f})")

    print(f"\n✅ Training finished. Best Loss: {best_loss:.6f}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"🔄 Loaded best model for visualization")

    plt.figure(figsize=(10, 4))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(test_loss_history, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('Autoencoder Training')
    plt.savefig('./output/ae/loss_curves.png', dpi=150)
    print("💾 Saved: loss_curves.png")

# Visualization 1: Reconstructions (10 original vs reconstructed)
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        for i in range(10):
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            axes[1, i].imshow(outputs[i].cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

        plt.tight_layout()
        plt.savefig('./output/ae/reconstructions.png', dpi=150)
        print("💾 Saved: reconstructions.png")
        break

# Visualization 2: Latent Space (t-SNE nếu latent_dim > 2, trực tiếp nếu latent_dim == 2)
model.eval()
latents = []
labels_list = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        z = model.encode(images)
        latents.append(z.cpu().numpy())
        labels_list.append(labels.numpy())

latents = np.concatenate(latents, axis=0)
labels_list = np.concatenate(labels_list, axis=0)

if latent_dim == 2:
    coords_2d = latents
    title = '2D Latent Space (actual 2D — no reduction)'
    xlabel, ylabel = 'Latent dim 1', 'Latent dim 2'
else:
    from sklearn.manifold import TSNE
    print(f"Running t-SNE: {latent_dim}D → 2D ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords_2d = tsne.fit_transform(latents)
    title = f't-SNE: {latent_dim}D Latent Space → 2D'
    xlabel, ylabel = 't-SNE dim 1', 't-SNE dim 2'

plt.figure(figsize=(10, 8))
scatter = plt.scatter(coords_2d[:, 0], coords_2d[:, 1],
                     c=labels_list, cmap='tab10',
                     alpha=0.6, s=5)
plt.colorbar(scatter, ticks=range(10))
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.grid(alpha=0.3)
plt.savefig('./output/ae/latent_space.png', dpi=150)
print("💾 Saved: latent_space.png")

# Visualization 3: Random Sampling (works for any latent_dim)
model.eval()
with torch.no_grad():
    # --- 3a: Uniform sampling (baseline garbage) ---
    z_uniform = torch.FloatTensor(10, latent_dim).uniform_(-3, 3).to(device)
    imgs_uniform = model.decode(z_uniform)

    # --- 3b: Empirical sampling: z ~ N(mu, std) per dim từ real latents ---
    mu_lat = latents.mean(axis=0)      # (latent_dim,)
    std_lat = latents.std(axis=0)      # (latent_dim,)
    z_empirical = torch.FloatTensor(
        np.random.normal(mu_lat, std_lat, size=(10, latent_dim))
    ).to(device)
    imgs_empirical = model.decode(z_empirical)

    # --- 3c: Interpolation giữa 2 ảnh thật ---
    # Lấy 2 ảnh từ test set (chọn digit 0 và digit 1 để dễ thấy)
    idx_a = np.where(labels_list == 0)[0][0]
    idx_b = np.where(labels_list == 1)[0][0]
    z_a = torch.FloatTensor(latents[idx_a]).unsqueeze(0).to(device)
    z_b = torch.FloatTensor(latents[idx_b]).unsqueeze(0).to(device)
    alphas = torch.linspace(0, 1, 10).to(device)
    z_interp = torch.stack([(1 - a) * z_a + a * z_b for a in alphas]).squeeze(1)
    imgs_interp = model.decode(z_interp)

    # --- 3d: PCA grid (2 principal components của latent space) ---
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(latents)
    # Grid 10x10 trong không gian PCA (±3 std theo mỗi PC)
    pc_std = latents @ pca.components_.T   # project all latents lên 2 PCs
    r1 = np.linspace(-3 * pc_std[:, 0].std(), 3 * pc_std[:, 0].std(), 10)
    r2 = np.linspace(-3 * pc_std[:, 1].std(), 3 * pc_std[:, 1].std(), 10)
    grid_imgs = []
    for v2 in reversed(r2):
        row_imgs = []
        for v1 in r1:
            z_pca = pca.mean_ + v1 * pca.components_[0] + v2 * pca.components_[1]
            z_t = torch.FloatTensor(z_pca).unsqueeze(0).to(device)
            img = model.decode(z_t)
            row_imgs.append(img[0].cpu().squeeze().numpy())
        grid_imgs.append(row_imgs)

    # Plot
    fig, axes = plt.subplots(4, 10, figsize=(15, 7))
    row_labels = [
        'Uniform\nN(-3,3)',
        'Empirical\nN(μ,σ)',
        'Interp\n0→1',
        'PCA grid\n(row 5/10)',
    ]
    for col in range(10):
        axes[0, col].imshow(imgs_uniform[col].cpu().squeeze(), cmap='gray')
        axes[1, col].imshow(imgs_empirical[col].cpu().squeeze(), cmap='gray')
        axes[2, col].imshow(imgs_interp[col].cpu().squeeze(), cmap='gray')
        axes[3, col].imshow(grid_imgs[5][col], cmap='gray')  # hàng giữa PCA grid
        for row in range(4):
            axes[row, col].axis('off')
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.05, 0.5, label, fontsize=8, ha='right', va='center', transform=axes[row, 0].transAxes)

    plt.suptitle(f'Random Sampling Strategies — latent_dim={latent_dim}', fontsize=11)
    plt.tight_layout()
    plt.savefig('./output/ae/random_sampling.png', dpi=150)
    print("💾 Saved: random_sampling.png")

print("\n✅ All done!")