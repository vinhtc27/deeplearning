import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

eval_only      = False
batch_size     = 256
epochs         = 30
lr             = 1e-3
weight_decay   = 1e-4
patch_size     = 7
num_patches    = (28 // patch_size) ** 2   # 16 patches (4×4 grid)
patch_dim      = patch_size * patch_size   # 49 pixels/patch
d_model        = 128
n_heads        = 4
d_ff           = 512
n_layers       = 4
n_classes      = 10
dropout        = 0.1
grid_size      = 28 // patch_size          # 4

model_path = "./output/vit/vit_best.pt"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


def patchify(x):
    # (B, 1, 28, 28) → (B, 16, 49): cắt 4×4 grid, mỗi patch 7×7 flatten
    B = x.size(0)
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return x.contiguous().view(B, num_patches, patch_dim)


def denorm(x):
    # tensor (1, 28, 28) normalized → numpy (28, 28) ∈ [0, 1]
    return (x.squeeze().cpu().numpy() * 0.3081 + 0.1307).clip(0, 1)


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.scale   = self.d_k ** -0.5
        # Q, K, V gộp thành 1 projection → 1 matmul thay vì 3
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                               # each (B, n_heads, N, d_k)
        attn    = (q @ k.transpose(-2, -1)) * self.scale      # (B, n_heads, N, N)
        attn    = attn.softmax(dim=-1)
        out     = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out), attn                            # attn: (B, n_heads, N, N)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))   # Pre-LN
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights                               # (B, n_heads, N+1, N+1)


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.drop        = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, patches, return_attn=False):
        B   = patches.size(0)
        x   = self.patch_embed(patches)               # (B, 16, d_model)
        cls = self.cls_token.expand(B, -1, -1)        # (B,  1, d_model)
        x   = torch.cat([cls, x], dim=1)              # (B, 17, d_model)
        x   = self.drop(x + self.pos_embed)

        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)                     # (B, n_heads, 17, 17)

        x      = self.norm(x)
        logits = self.head(x[:, 0])                    # CLS token → classify

        if return_attn:
            return logits, attn_maps
        return logits


model = ViT().to(device)
print(model)
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

criterion = nn.CrossEntropyLoss()

if eval_only:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Set eval_only = False to train first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")
else:
    best_acc           = 0.0
    optimizer          = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler          = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loss_history = []
    train_acc_history  = []
    test_acc_history   = []

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss    = 0.0
        train_correct = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            patches      = patchify(data)
            optimizer.zero_grad()
            logits = model(patches)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss    += loss.item() * data.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
        scheduler.step()

        train_loss   /= len(train_loader.dataset)
        train_acc     = train_correct / len(train_loader.dataset) * 100
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        model.train(False)
        test_correct = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                logits       = model(patchify(data))
                test_correct += (logits.argmax(1) == labels).sum().item()
        test_acc = test_correct / len(test_loader.dataset) * 100
        test_acc_history.append(test_acc)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {train_loss:.4f}  Train: {train_acc:.2f}%  Test: {test_acc:.2f}%  Time: {epoch_time:.2f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)
            print(f"🔥 New best model saved! ({best_acc:.2f}%)")

    print(f"\n✅ Training finished. Best Test Accuracy: {best_acc:.2f}%")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("🔄 Loaded best model for visualization")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_loss_history, color='steelblue', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training Loss'); axes[0].grid(alpha=0.3)

    axes[1].plot(train_acc_history, label='Train', color='steelblue', linewidth=2)
    axes[1].plot(test_acc_history,  label='Test',  color='tomato',    linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 100)

    plt.suptitle('ViT Training — MNIST', fontsize=12)
    plt.tight_layout()
    plt.savefig('./output/vit/training_curves.png', dpi=150)
    print("💾 Saved: training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 1: Accuracy & confusion matrix trên test set
# ─────────────────────────────────────────────────────────────────────────────
model.train(False)
all_preds, all_labels = [], []
with torch.no_grad():
    for data, labels in test_loader:
        preds = model(patchify(data.to(device))).argmax(1).cpu()
        all_preds.append(preds); all_labels.append(labels)

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
accuracy   = (all_preds == all_labels).mean() * 100

conf_matrix = np.zeros((10, 10), dtype=int)
for t, p in zip(all_labels, all_preds):
    conf_matrix[t, p] += 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"ViT — MNIST Test Set  ·  Accuracy: {accuracy:.2f}%", fontsize=12)

class_acc = [conf_matrix[c, c] / conf_matrix[c].sum() * 100 for c in range(10)]
axes[0].bar(range(10), class_acc, color='steelblue', edgecolor='white')
axes[0].set_xticks(range(10)); axes[0].set_xlabel('Digit Class'); axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Per-class Accuracy'); axes[0].set_ylim(80, 100); axes[0].grid(axis='y', alpha=0.3)
for c, acc in enumerate(class_acc):
    axes[0].text(c, acc + 0.05, f"{acc:.1f}", ha='center', va='bottom', fontsize=7)

im = axes[1].imshow(conf_matrix, cmap='Blues')
axes[1].set_xticks(range(10)); axes[1].set_yticks(range(10))
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix')
for r in range(10):
    for c in range(10):
        val   = conf_matrix[r, c]
        color = 'white' if val > conf_matrix[r].sum() * 0.5 else 'black'
        axes[1].text(c, r, str(val), ha='center', va='center', fontsize=7, color=color)
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('./output/vit/accuracy_confusion.png', dpi=150)
print("💾 Saved: accuracy_confusion.png")


# ─────────────────────────────────────────────────────────────────────────────
# Collect 1 correctly-predicted sample per digit class (dùng cho Viz 2 & 3)
# ─────────────────────────────────────────────────────────────────────────────
model.train(False)
class_samples = {}   # {digit: (img_tensor, attn_per_layer)}

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        logits, attn_maps = model(patchify(data), return_attn=True)
        preds = logits.argmax(1)

        for i in range(data.size(0)):
            lbl = labels[i].item()
            if lbl not in class_samples and preds[i].item() == lbl:
                class_samples[lbl] = (
                    data[i].cpu(),
                    [a[i].cpu() for a in attn_maps]   # list of (n_heads, 17, 17)
                )
        if len(class_samples) == 10:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 2: CLS attention heatmap — 1 sample mỗi chữ số
#   Row trên: ảnh gốc | Row dưới: ảnh + attention overlay (last layer, mean heads)
#   Thấy model "nhìn vào đâu" khi classify từng chữ số
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 10, figsize=(20, 4.5))
fig.suptitle("CLS Attention Heatmap  ·  Last Layer, Mean over Heads\n"
             "(đỏ = vùng model chú ý nhiều nhất khi classify)", fontsize=11)

for digit in range(10):
    img_tensor, attn = class_samples[digit]
    img = denorm(img_tensor)

    # attn[-1]: (n_heads, 17, 17); CLS→patches: [:, 0, 1:]
    cls_attn = attn[-1][:, 0, 1:].mean(0).numpy()    # (16,) mean over heads
    heat     = cls_attn.reshape(grid_size, grid_size)
    heat     = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_up  = heat.repeat(patch_size, axis=0).repeat(patch_size, axis=1)

    axes[0, digit].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, digit].set_title(f"Digit {digit}", fontsize=9)
    axes[0, digit].axis('off')

    axes[1, digit].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[1, digit].imshow(heat_up, cmap='Reds', alpha=0.55, vmin=0, vmax=1)
    axes[1, digit].axis('off')

axes[0, 0].set_ylabel("Original", fontsize=8, rotation=0, labelpad=35)
axes[1, 0].set_ylabel("Attention", fontsize=8, rotation=0, labelpad=35)

plt.tight_layout()
plt.savefig('./output/vit/cls_attention_heatmap.png', dpi=150)
print("💾 Saved: cls_attention_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 3: Multi-head attention — tất cả layer × head cho 1 ảnh
#   Grid: n_layers rows × n_heads cols; mỗi ô = CLS attention 1 head 1 layer
#   Thấy mỗi head học đặc trưng không gian khác nhau, deepens qua từng layer
# ─────────────────────────────────────────────────────────────────────────────
sample_digit = 3
sample_img_t, sample_attn = class_samples[sample_digit]
sample_img   = denorm(sample_img_t)

fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 2.8, n_layers * 2.8))
fig.suptitle(f"Multi-head Attention per Layer  ·  Digit: {sample_digit}\n"
             f"(mỗi head học một đặc trưng không gian khác nhau)", fontsize=11)

for layer in range(n_layers):
    for head in range(n_heads):
        cls_attn_head = sample_attn[layer][head, 0, 1:].numpy()   # (16,)
        heat     = cls_attn_head.reshape(grid_size, grid_size)
        heat     = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_up  = heat.repeat(patch_size, axis=0).repeat(patch_size, axis=1)

        ax = axes[layer, head]
        ax.imshow(sample_img, cmap='gray', vmin=0, vmax=1)
        ax.imshow(heat_up, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        ax.axis('off')
        if layer == 0:
            ax.set_title(f"Head {head + 1}", fontsize=10)
        if head == 0:
            ax.set_ylabel(f"Layer {layer + 1}", fontsize=9)

plt.tight_layout()
plt.savefig('./output/vit/multihead_attention.png', dpi=150)
print("💾 Saved: multihead_attention.png")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 4: Attention entropy across layers per digit class
#   Entropy = -Σ p·log(p) của CLS attention distribution trên 16 patch
#   Low entropy  → attention tập trung (model biết nhìn đâu)
#   High entropy → attention phân tán (model chưa focus)
#   Reference: uniform over 16 patches → entropy = log(16) ≈ 2.77 nats
# ─────────────────────────────────────────────────────────────────────────────
model.train(False)
entropy_per_class = np.zeros((10, n_layers))
count_per_class   = np.zeros(10)

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        _, attn_maps = model(patchify(data), return_attn=True)

        for layer_idx in range(n_layers):
            # CLS → patch attention: mean over heads → (B, 16)
            cls_attn = attn_maps[layer_idx][:, :, 0, 1:].mean(1).cpu().numpy()
            cls_attn = np.clip(cls_attn, 1e-9, 1.0)
            entropy  = -(cls_attn * np.log(cls_attn)).sum(axis=1)   # (B,)

            lbl_np = labels.cpu().numpy()
            for cls in range(10):
                mask = (lbl_np == cls)
                if mask.any():
                    entropy_per_class[cls, layer_idx] += entropy[mask].sum()

        count_per_class += np.array([(labels.cpu().numpy() == c).sum() for c in range(10)])

entropy_per_class /= count_per_class[:, None]

fig, ax = plt.subplots(figsize=(9, 5))
cmap10  = plt.cm.tab10
for cls in range(10):
    ax.plot(range(1, n_layers + 1), entropy_per_class[cls],
            label=str(cls), color=cmap10(cls / 10), linewidth=2, marker='o', markersize=6)

max_entropy = math.log(num_patches)
ax.axhline(max_entropy, linestyle='--', color='gray', linewidth=1)
ax.text(n_layers, max_entropy + 0.02, f"Uniform  max = {max_entropy:.2f}",
        va='bottom', ha='right', fontsize=8, color='gray')

ax.set_xlabel("Layer")
ax.set_ylabel("Mean Attention Entropy (nats)")
ax.set_title("Attention Entropy across Layers per Digit Class\n"
             "(thấp = tập trung;  cao = phân tán đều trên 16 patch)")
ax.set_xticks(range(1, n_layers + 1))
ax.legend(title="Digit", ncol=2, fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./output/vit/attention_entropy.png', dpi=150)
print("💾 Saved: attention_entropy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 5: Progressive patch masking
#   Sort 16 patch theo CLS attention weight (thấp → cao)
#   Mask dần từng patch, đo confidence class đúng
#   Xóa patch ít-attention → confidence giảm chậm
#   Xóa patch nhiều-attention → confidence sụp đổ nhanh
# ─────────────────────────────────────────────────────────────────────────────
mask_digit   = 3
mask_img_t, mask_attn = class_samples[mask_digit]
mask_img     = denorm(mask_img_t)

cls_attn_last = mask_attn[-1][:, 0, 1:].mean(0).numpy()   # (16,)
patch_order   = np.argsort(cls_attn_last)                   # least → most attended

mask_steps = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
n_cols     = len(mask_steps)

gray_val = (0.5 - 0.1307) / 0.3081

confidences = []
masked_imgs = []

model.train(False)
with torch.no_grad():
    for n_masked in mask_steps:
        img_t = mask_img_t.clone()

        for p_idx in patch_order[:n_masked]:
            r = (p_idx // grid_size) * patch_size
            c = (p_idx %  grid_size) * patch_size
            img_t[0, r:r+patch_size, c:c+patch_size] = gray_val

        masked_imgs.append(denorm(img_t))

        patches = patchify(img_t.unsqueeze(0).to(device))
        logits  = model(patches)
        probs   = logits.softmax(dim=-1).squeeze()
        confidences.append(probs[mask_digit].item() * 100)

fig = plt.figure(figsize=(n_cols * 1.8, 4.5))
fig.suptitle(f"Progressive Patch Masking  ·  Digit: {mask_digit}\n"
             f"(xóa dần patch ít-attention trước → confidence giảm chậm; "
             f"xóa patch quan trọng → sụp đổ nhanh)", fontsize=10)

gs = fig.add_gridspec(2, n_cols, height_ratios=[1.8, 1], hspace=0.35)

for col, (n_masked, img_m) in enumerate(zip(mask_steps, masked_imgs)):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img_m, cmap='gray', vmin=0, vmax=1)

    overlay = np.zeros((28, 28, 4))
    for p_idx in range(num_patches):
        r = (p_idx // grid_size) * patch_size
        c = (p_idx %  grid_size) * patch_size
        if p_idx in patch_order[:n_masked]:
            overlay[r:r+patch_size, c:c+patch_size] = [0.5, 0.5, 0.5, 0.35]
        else:
            overlay[r:r+patch_size, c:c+patch_size] = [0.1, 0.7, 0.1, 0.15]
    ax.imshow(overlay)
    ax.set_title(f"mask={n_masked}\n{confidences[col]:.1f}%", fontsize=7.5)
    ax.axis('off')

ax_line = fig.add_subplot(gs[1, :])
ax_line.plot(mask_steps, confidences, marker='o', linewidth=2,
             markersize=5, color='steelblue')
ax_line.axhline(10, linestyle='--', color='gray', linewidth=1)
ax_line.text(mask_steps[-1], 11, "random (10%)", ha='right', fontsize=7, color='gray')
ax_line.set_xlabel("Số patch bị mask (ít attention → nhiều attention)")
ax_line.set_ylabel(f"Confidence digit {mask_digit} (%)")
ax_line.set_xticks(mask_steps)
ax_line.set_ylim(0, 105)
ax_line.grid(alpha=0.3)

plt.savefig('./output/vit/patch_masking.png', dpi=150)
print("💾 Saved: patch_masking.png")

print("\n✅ All done!")
