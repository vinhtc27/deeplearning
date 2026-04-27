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
model_path = "./output/vae/vae_best.pt"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder backbone: học feature chung từ ảnh đầu vào (784 → 256 → 64)
        # Output của backbone sẽ được đọc bởi 2 head song song bên dưới
        self.encoder_backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # Hai head song song — thay vì AE chỉ có 1 Linear ra z trực tiếp,
        # VAE output 2 vector để tham số hóa phân phối posterior q_φ(z|x) = N(μ, diag(σ²))
        # fc_mu  → μ : trung tâm của phân phối posterior
        # fc_logvar → log σ² : dùng log thay vì σ² trực tiếp vì log không bị ràng buộc dương,
        #             mạng optimize dễ hơn; σ sẽ được khôi phục bằng exp(0.5 * logvar)
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder: giữ nguyên so với AE — nhận z, giải mã ra ảnh
        # Sigmoid ở cuối đảm bảo output ∈ [0,1], khớp với pixel MNIST,
        # cho phép dùng BCE loss (Bernoulli likelihood) thay vì MSE (Gaussian likelihood)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        # Reparameterization Trick — giải quyết vấn đề gradient không chạy qua bước sampling
        #
        # Vấn đề gốc: z ~ N(μ, σ²) là phép sampling ngẫu nhiên → không differentiable
        #             → gradient bị chặn, không backprop được về encoder
        #
        # Giải pháp: tách randomness ra một biến ngoài ε không có tham số:
        #   z = μ + σ ⊙ ε,   ε ~ N(0, I)
        # z vẫn có phân phối N(μ, σ²), nhưng bây giờ là hàm deterministic của (μ, σ)
        # → gradient chạy qua z về μ và σ → về encoder bình thường
        #
        # Chỉ sample khi training; lúc eval dùng μ để reconstruction ổn định, deterministic
        if self.training:
            std = torch.exp(0.5 * logvar)   # σ = exp(log σ² / 2)
            eps = torch.randn_like(std)      # ε ~ N(0, I), shape giống std
            return mu + eps * std            # z = μ + ε · σ
        else:
            return mu

    def forward(self, x):
        # Khác AE: forward trả về (x_recon, mu, logvar) thay vì chỉ x_recon
        # vì loss function cần mu và logvar để tính KL — không thể tính KL chỉ từ z
        h       = self.encoder_backbone(x)
        mu      = self.fc_mu(h)
        logvar  = self.fc_logvar(h)
        z       = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z).view(-1, 1, 28, 28)
        return x_recon, mu, logvar

    def encode(self, x):
        # Trả về (μ, log σ²) — dùng cho visualization cần cả hai (latent space, KL per dim)
        h = self.encoder_backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)


model = VAE(latent_dim=latent_dim).to(device)
print(model)
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")


# ════════════════════════════════════════════════════════════════════════════
# Loss function — ELBO (Evidence Lower BOund)
# ════════════════════════════════════════════════════════════════════════════
def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    # VAE tối đa hóa ELBO ↔ tối thiểu hóa loss sau:
    #
    #   L = Recon Loss  +  β · KL Loss
    #     = -E[log p_θ(x|z)]  +  β · KL(q_φ(z|x) ‖ N(0,I))
    #
    # Hai lực đối lập:
    #   Recon Loss muốn z encode nhiều thông tin nhất có thể về x
    #   KL Loss muốn z "quên" x và trở về prior N(0,I)
    # → Cân bằng tạo latent space mịn, liên tục, có thể sample được

    # --- Reconstruction Loss ---
    # Pixel MNIST ∈ [0,1] → mô hình hóa là Bernoulli → likelihood = BCE
    # Dùng MSE sẽ tương ứng với Gaussian likelihood — sai model, recon kém hơn
    #
    # reduction='sum' rồi tự chia batch thay vì reduction='mean' vì:
    #   'mean' chia cho (B × H × W), còn KL chỉ chia cho B
    #   → hai loss ở hai scale khác nhau, tỷ lệ KL/Recon bị méo
    #   → dùng 'sum'/B đồng bộ scale của cả hai term
    recon_loss = nn.functional.binary_cross_entropy(
        x_recon, x, reduction='sum'
    ) / x.size(0)

    # --- KL Divergence Loss — dạng closed-form (không cần Monte Carlo) ---
    # Vì cả posterior q_φ(z|x) = N(μ, σ²I) và prior p(z) = N(0,I) đều là Gaussian,
    # KL tính được analytic:
    #
    #   KL(N(μ,σ²) ‖ N(0,I)) = -½ Σ_j (1 + log σ²_j - μ²_j - σ²_j)
    #
    # Trong code: logvar = log σ²,  logvar.exp() = σ²
    # torch.sum chạy trên cả batch và latent dim, chia B để lấy mean theo batch
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # beta=1.0 là VAE chuẩn; beta > 1 là β-VAE — tăng áp lực KL để học disentangled repr.
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════
if eval_only:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Set eval_only = False to train first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")
else:
    best_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history  = []
    test_loss_history   = []

    # Track recon và KL riêng để phát hiện posterior collapse sớm
    # Posterior collapse: KL → 0 (encoder bỏ qua input, q_φ(z|x) ≈ p(z))
    # trong khi recon loss vẫn cao → mạng không học được gì
    # Nếu chỉ xem total loss thì không phát hiện được hiện tượng này
    train_recon_history = []
    train_kl_history    = []
    test_recon_history  = []
    test_kl_history     = []

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss = train_recon = train_kl = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            # Unpack 3 giá trị — gradient flow: loss → x_recon → z → (mu, logvar) → encoder
            # Reparameterization trick đảm bảo gradient không bị chặn tại bước sampling z
            x_recon, mu, logvar = model(images)
            loss, recon, kl = vae_loss(x_recon, images, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = images.size(0)
            train_loss  += loss.item()  * n
            train_recon += recon.item() * n
            train_kl    += kl.item()    * n

        N = len(train_loader.dataset)
        train_loss /= N; train_recon /= N; train_kl /= N
        train_loss_history.append(train_loss)
        train_recon_history.append(train_recon)
        train_kl_history.append(train_kl)

        model.eval()
        test_loss = test_recon = test_kl = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                x_recon, mu, logvar = model(images)
                loss, recon, kl = vae_loss(x_recon, images, mu, logvar)
                n = images.size(0)
                test_loss  += loss.item()  * n
                test_recon += recon.item() * n
                test_kl    += kl.item()    * n

        N = len(test_loader.dataset)
        test_loss /= N; test_recon /= N; test_kl /= N
        test_loss_history.append(test_loss)
        test_recon_history.append(test_recon)
        test_kl_history.append(test_kl)

        epoch_time = time.time() - start_time

        # Log recon và KL riêng — nếu KL → 0 mà recon vẫn cao = posterior collapse
        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"Loss: {test_loss:.2f}  "
              f"Recon: {test_recon:.2f}  "
              f"KL: {test_kl:.2f}  "
              f"Time: {epoch_time:.2f}s")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f"  🔥 New best model saved! ({best_loss:.4f})")

    print(f"\n✅ Training finished. Best Loss: {best_loss:.4f}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"🔄 Loaded best model for visualization")

    # Plot 3 subplots riêng: total ELBO / recon / KL
    # Xem KL curve: nếu KL tăng dần rồi ổn định = bình thường
    #               nếu KL → 0 sớm = posterior collapse, cần giảm beta hoặc anneal KL
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_loss_history, label='Train')
    axes[0].plot(test_loss_history,  label='Test')
    axes[0].set_title('Total ELBO Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(train_recon_history, label='Train')
    axes[1].plot(test_recon_history,  label='Test')
    axes[1].set_title('Reconstruction Loss (BCE)'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(train_kl_history, label='Train')
    axes[2].plot(test_kl_history,  label='Test')
    axes[2].set_title('KL Divergence'); axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle('VAE Training Curves', fontsize=12)
    plt.tight_layout()
    plt.savefig('./output/vae/loss_curves.png', dpi=150)
    print("💾 Saved: loss_curves.png")


# ════════════════════════════════════════════════════════════════════════════
# Visualization 1 — Reconstruction
# ════════════════════════════════════════════════════════════════════════════

# Lúc eval, reparameterize() trả về μ (không sample) → reconstruction deterministic
# So sánh original vs reconstructed để đánh giá recon loss trực quan
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs, _, _ = model(images)   # mu, logvar không cần cho visualization này

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
        plt.savefig('./output/vae/reconstructions.png', dpi=150)
        print("💾 Saved: reconstructions.png")
        break


# ════════════════════════════════════════════════════════════════════════════
# Visualization 2 — Latent Space (t-SNE hoặc 2D trực tiếp)
# ════════════════════════════════════════════════════════════════════════════

# Thu thập μ (không phải z sample) của toàn bộ test set để plot latent space
# μ là đại diện ổn định nhất của x trong latent space — ít noise hơn z sample
# Nếu VAE học tốt, các class sẽ tạo thành cluster riêng nhưng vẫn chồng lấp nhau
# (khác AE: cluster tách biệt hoàn toàn, không interpolate được)
model.eval()
latents     = []
labels_list = []
all_mu      = []   # lưu mu tensor để tính KL per dim ở visualization 2b
all_logvar  = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        mu, logvar = model.encode(images)
        all_mu.append(mu.cpu())
        all_logvar.append(logvar.cpu())
        latents.append(mu.cpu().numpy())
        labels_list.append(labels.numpy())

latents     = np.concatenate(latents, axis=0)
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
    title  = f't-SNE: {latent_dim}D Latent Space → 2D'
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
plt.savefig('./output/vae/latent_space.png', dpi=150)
print("💾 Saved: latent_space.png")


# ════════════════════════════════════════════════════════════════════════════
# Visualization 2b — KL per Dimension (Active Units)
# ════════════════════════════════════════════════════════════════════════════

# Metric chẩn đoán quan trọng nhất của VAE: mỗi chiều latent đóng góp bao nhiêu KL?
#
# Một chiều j là "active" nếu KL_j > threshold, tức encoder đang thực sự
# encode thông tin vào chiều đó (μ_j và σ_j khác xa prior N(0,1))
#
# Chiều "collapsed" có KL_j ≈ 0: q_φ(z_j|x) ≈ N(0,1) với mọi x
# → encoder bỏ qua chiều đó hoàn toàn, decoder cũng không dùng
#
# Với latent_dim=32 trên MNIST: kỳ vọng ~10–15 chiều active vì MNIST chỉ
# có ~10 class, không đủ thông tin để lấp đầy 32 chiều
mu_tensor     = torch.cat(all_mu)      # (N, latent_dim)
logvar_tensor = torch.cat(all_logvar)  # (N, latent_dim)

kl_per_dim = -0.5 * (1 + logvar_tensor - mu_tensor.pow(2) - logvar_tensor.exp())
kl_per_dim = kl_per_dim.mean(0).numpy()   # mean theo batch → (latent_dim,)

KL_THRESHOLD = 0.1
active_units = (kl_per_dim > KL_THRESHOLD).sum()
print(f"Active units: {active_units}/{latent_dim}")

sorted_idx = np.argsort(kl_per_dim)[::-1]
colors = ['#2563eb' if kl_per_dim[i] > KL_THRESHOLD else '#d1d5db' for i in sorted_idx]

plt.figure(figsize=(14, 4))
plt.bar(range(latent_dim), kl_per_dim[sorted_idx], color=colors)
plt.axhline(KL_THRESHOLD, color='red', linestyle='--', linewidth=1,
            label=f'threshold = {KL_THRESHOLD}')
plt.xlabel('Latent dimension (sorted by KL)')
plt.ylabel('Mean KL divergence')
plt.title(f'KL per dimension — Active units: {active_units}/{latent_dim}')
plt.legend(); plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./output/vae/kl_per_dim.png', dpi=150)
print("💾 Saved: kl_per_dim.png")


# ════════════════════════════════════════════════════════════════════════════
# Visualization 3 — Sampling, Interpolation, PCA Grid
# ════════════════════════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():

    # Prior sampling — điểm khác biệt cốt lõi so với AE
    # AE không có prior → không biết sample từ đâu (phải dùng empirical mean/range)
    # VAE có prior N(0,I) → sample z ~ N(0,I) và decode trực tiếp
    # KL loss đảm bảo toàn bộ vùng N(0,I) đều được decoder "phủ sóng" có nghĩa
    z_prior    = torch.randn(10, latent_dim).to(device)
    imgs_prior = model.decode(z_prior)

    # Linear interpolation giữa 2 điểm trong latent space
    # z_interp = (1-α)·z_a + α·z_b,  α ∈ [0,1]
    # Vì latent space của VAE liên tục (các posterior blob chồng lên nhau),
    # điểm trung gian decode ra ảnh trung gian có nghĩa (ví dụ: chữ số giữa 0 và 1)
    # AE không đảm bảo điều này vì latent space có vùng trống giữa các cluster
    idx_a = np.where(labels_list == 0)[0][0]
    idx_b = np.where(labels_list == 1)[0][0]
    z_a = torch.FloatTensor(latents[idx_a]).unsqueeze(0).to(device)
    z_b = torch.FloatTensor(latents[idx_b]).unsqueeze(0).to(device)
    alphas = torch.linspace(0, 1, 10).to(device)
    z_interp = torch.stack([(1 - a) * z_a + a * z_b for a in alphas]).squeeze(1)
    imgs_interp = model.decode(z_interp)

    # PCA grid — traverse latent space dọc theo 2 principal component
    # PC1, PC2 là 2 hướng biến thiên nhiều nhất trong latent space
    # Grid này cho thấy latent space tổ chức thông tin như thế nào một cách có hệ thống
    # r1, r2 lấy trong khoảng ±3 std của data thực (tránh sample quá xa prior)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(latents)
    pc_scores = pca.transform(latents)   # (N, 2) — centered scores
    r1 = np.linspace(-3 * pc_scores[:, 0].std(), 3 * pc_scores[:, 0].std(), 10)
    r2 = np.linspace(-3 * pc_scores[:, 1].std(), 3 * pc_scores[:, 1].std(), 10)
    grid_imgs = []
    for v2 in reversed(r2):
        row_imgs = []
        for v1 in r1:
            z_pca = pca.mean_ + v1 * pca.components_[0] + v2 * pca.components_[1]
            z_t = torch.FloatTensor(z_pca).unsqueeze(0).to(device)
            row_imgs.append(model.decode(z_t)[0].cpu().squeeze().numpy())
        grid_imgs.append(row_imgs)

    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    row_labels = [
        'Prior\nN(0,I)',
        'Interp\n0→1',
        'PCA grid\n(row 6/10)',
    ]
    for col in range(10):
        axes[0, col].imshow(imgs_prior[col].cpu().squeeze(),  cmap='gray')
        axes[1, col].imshow(imgs_interp[col].cpu().squeeze(), cmap='gray')
        axes[2, col].imshow(grid_imgs[5][col],                cmap='gray')
        for row in range(3):
            axes[row, col].axis('off')
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.05, 0.5, label, fontsize=8, ha='right', va='center',
                          transform=axes[row, 0].transAxes)

    plt.suptitle(f'Sampling from VAE — latent_dim={latent_dim}', fontsize=11)
    plt.tight_layout()
    plt.savefig('./output/vae/random_sampling.png', dpi=150)
    print("💾 Saved: random_sampling.png")

print("\n✅ All done!")