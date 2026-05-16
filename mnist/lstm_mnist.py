import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math
import time
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

eval_only   = True
batch_size  = 64
epochs      = 20
lr          = 1e-3
patch_size  = 7
num_patches = (28 // patch_size) ** 2   # 16 patches (4×4 grid)
input_size  = patch_size * patch_size   # 49 pixels/patch
hidden_size = 256
sample_idx  = 1   # index ảnh dùng cho viz step-by-step (Visualization 2)

model_path = "./output/lstm/lstm_best.pt"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


def patchify(x):
    # (B, 1, 28, 28) → (B, 16, 49): cắt 4×4 grid, mỗi patch 7×7 flatten
    B = x.size(0)
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # unfold appends new dim cuối → (B, 1, 4, 4, 7, 7)
    return x.contiguous().view(B, num_patches, input_size)


def unpatchify(patches):
    # (B, 16, 49) → (B, 1, 28, 28): ngược lại patchify
    B    = patches.size(0)
    grid = 28 // patch_size
    p    = patches.view(B, 1, grid, grid, patch_size, patch_size)
    # permute để ghép (grid_row, patch_row) → H và (grid_col, patch_col) → W
    p    = p.permute(0, 1, 2, 4, 3, 5).contiguous()
    return p.view(B, 1, 28, 28)


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # [Wf | Wi | Wg | Wo] gộp thành 1 linear → 1 matmul thay vì 4
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        f_raw, i_raw, g_raw, o_raw = self.linear(
            torch.cat([x, h_prev], dim=1)
        ).chunk(4, dim=1)
        f = torch.sigmoid(f_raw)
        i = torch.sigmoid(i_raw)
        g = torch.tanh(g_raw)
        o = torch.sigmoid(o_raw)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class PatchLSTM(nn.Module):
    # Task: đọc patch t → predict pixel của patch t+1
    # Train  : teacher forcing (feed actual patches)
    # Predict: autoregressive (feed predicted patch làm input tiếp theo)
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell   = LSTMCell(input_size, hidden_size)
        self.head        = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()   # pixel values ∈ [0, 1]
        )

    def forward(self, patches):
        # Teacher forcing: feed patch 0..14, predict 1..15
        # patches: (B, 16, 49) → returns (B, 15, 49)
        B = patches.size(0)
        h = torch.zeros(B, self.hidden_size, device=patches.device)
        c = torch.zeros(B, self.hidden_size, device=patches.device)
        preds = []
        for t in range(num_patches - 1):
            h, c = self.lstm_cell(patches[:, t], h, c)
            preds.append(self.head(h))
        return torch.stack(preds, dim=1)   # (B, 15, 49)

    def predict_autoregressive(self, patches, num_given):
        # Warm up với num_given patch ground truth, sau đó tự predict:
        # output bước t trở thành input bước t+1 — không dùng ground truth nữa
        B = patches.size(0)
        h = torch.zeros(B, self.hidden_size, device=patches.device)
        c = torch.zeros(B, self.hidden_size, device=patches.device)

        for t in range(num_given):
            h, c = self.lstm_cell(patches[:, t], h, c)

        pred  = self.head(h)   # predict patch num_given
        preds = [pred]
        for _ in range(num_patches - num_given - 1):
            h, c = self.lstm_cell(pred, h, c)
            pred = self.head(h)
            preds.append(pred)

        return torch.stack(preds, dim=1)   # (B, num_patches - num_given, 49)


model = PatchLSTM(input_size=input_size, hidden_size=hidden_size).to(device)
print(model)
total_params     = sum(p.numel() for p in model.parameters())
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
    best_loss          = float('inf')
    optimizer          = optim.Adam(model.parameters(), lr=lr)
    train_loss_history = []
    test_loss_history  = []

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data    = data.to(device)
            patches = patchify(data)
            optimizer.zero_grad()
            preds   = model(patches)
            loss    = criterion(preds, patches[:, 1:])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                data    = data.to(device)
                patches = patchify(data)
                preds   = model(patches)
                test_loss += criterion(preds, patches[:, 1:]).item() * data.size(0)
        test_loss /= len(test_loader.dataset)
        test_loss_history.append(test_loss)
        psnr       = 10 * math.log10(1.0 / test_loss)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] Train: {train_loss:.6f}  Test: {test_loss:.6f}  PSNR: {psnr:.2f}dB  Time: {epoch_time:.2f}s")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f"🔥 New best model saved! ({best_loss:.6f})")

    print(f"\n✅ Training finished. Best Loss: {best_loss:.6f}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("🔄 Loaded best model for visualization")

    plt.figure(figsize=(10, 4))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(test_loss_history,  label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('PatchLSTM Training')
    plt.tight_layout()
    plt.savefig('./output/lstm/loss_curves.png', dpi=150)
    print("💾 Saved: loss_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 1: Prediction error analysis
#   Left  — 4×4 heatmap: vị trí không gian nào khó predict nhất
#   Right — line chart  : lỗi theo thứ tự đọc patch (raster order)
#   Cả hai dùng teacher forcing → lỗi từng vị trí độc lập, không cộng dồn
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
patch_errors = torch.zeros(num_patches)   # position 0 = not predicted → stays 0

with torch.no_grad():
    n_samples = 0
    for data, _ in test_loader:
        data    = data.to(device)
        B       = data.size(0)
        patches = patchify(data)
        preds   = model(patches)
        errors  = ((preds - patches[:, 1:]) ** 2).mean(dim=(0, 2))  # (15,)
        patch_errors[1:] += errors.cpu() * B
        n_samples += B

patch_errors[1:] /= n_samples
grid_size  = 28 // patch_size
error_grid = patch_errors.numpy().reshape(grid_size, grid_size)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Prediction Error  ·  Teacher Forcing (mỗi bước dùng ground truth làm input)", fontsize=11)

# heatmap: không gian — ô nào đỏ = khó predict
im = axes[0].imshow(error_grid, cmap='Reds', interpolation='nearest')
for r in range(grid_size):
    for c in range(grid_size):
        val   = error_grid[r, c]
        color = 'white' if val > error_grid.max() * 0.55 else 'black'
        label = "given" if r == 0 and c == 0 else f"{val:.4f}"
        axes[0].text(c, r, label, ha='center', va='center', fontsize=8, color=color)
axes[0].set_title("MSE per patch position\n(đỏ đậm = khó predict hơn)")
axes[0].set_xlabel("Patch col")
axes[0].set_ylabel("Patch row")
plt.colorbar(im, ax=axes[0])

# line chart: temporal — lỗi thay đổi theo thứ tự đọc
axes[1].plot(range(1, num_patches), patch_errors[1:].numpy(),
             marker='o', linewidth=2, markersize=5, color='steelblue')
axes[1].set_xlabel("Patch index (raster order: 0=top-left → 15=bottom-right)")
axes[1].set_ylabel("Average MSE")
axes[1].set_title("Lỗi theo thứ tự đọc patch\n(trái→phải, trên→dưới)")
axes[1].set_xticks(range(1, num_patches))
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./output/lstm/error_analysis.png', dpi=150)
print("💾 Saved: error_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# Visualization 2: Single image — progressive completion
#   ảnh tại từng mốc num_given khác nhau (given=1..15 + original)
#   xanh = patch cho trước (ground truth), đỏ = patch model tự predict
# ─────────────────────────────────────────────────────────────────────────────
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        sample_img   = images[sample_idx:sample_idx+1].to(device)  # (1, 1, 28, 28)
        sample_label = labels[sample_idx].item()
        break

# --- Part A: Progressive completion ---
given_stages = [1, 2, 3, 4, 6, 8, 10, 12, 14, 15]
n_cols       = len(given_stages) + 1   # +1 for original at end

fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 1.6, 2.6))
fig.suptitle(f"Progressive Completion  ·  Digit: {sample_label}  "
             f"(xanh = given, đỏ = predicted)", fontsize=10)

for col, ng in enumerate(given_stages):
    with torch.no_grad():
        patches      = patchify(sample_img)
        pred_patches = model.predict_autoregressive(patches, ng)
        combined     = patches.clone()
        combined[:, ng:] = pred_patches
        img_pred = unpatchify(combined).squeeze().cpu().numpy()   # (28, 28)

    ax = axes[col]
    ax.imshow(img_pred, cmap='gray', vmin=0, vmax=1)

    # color overlay: xanh = given, đỏ = predicted
    overlay = np.zeros((28, 28, 4))
    for p in range(num_patches):
        r = (p // grid_size) * patch_size
        c = (p % grid_size) * patch_size
        color = [0, 0.8, 0, 0.25] if p < ng else [1, 0.1, 0.1, 0.25]
        overlay[r:r+patch_size, c:c+patch_size] = color
    ax.imshow(overlay)
    ax.set_title(f"given={ng}", fontsize=8)
    ax.axis('off')

axes[-1].imshow(sample_img.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[-1].set_title("Original", fontsize=8, fontweight='bold')
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('./output/lstm/step_by_step.png', dpi=150)
print("💾 Saved: step_by_step.png")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 3: LSTM internals — gate dynamics & memory states (sample image)
#   Top row  — 4 gate charts (f/i/g/o): mean ± std over 256 neurons, per step
#              thấy "nhịp" của từng gate khi LSTM đọc qua ảnh từng patch
#   Middle   — h_t heatmap: hidden state (short-term output, dao động nhiều)
#   Bottom   — c_t heatmap: cell state (long-term memory, tích lũy chậm)
# ─────────────────────────────────────────────────────────────────────────────
def collect_lstm_states(patches_1):
    # patches_1: (1, 16, 49) → returns gate + state arrays, each (16, hidden_size)
    h = torch.zeros(1, hidden_size, device=device)
    c = torch.zeros(1, hidden_size, device=device)
    out = {k: [] for k in ('f', 'i', 'g', 'o', 'h', 'c')}

    for t in range(num_patches):
        combined = torch.cat([patches_1[:, t], h], dim=1)
        f_raw, i_raw, g_raw, o_raw = model.lstm_cell.linear(combined).chunk(4, dim=1)
        f   = torch.sigmoid(f_raw);  i_g = torch.sigmoid(i_raw)
        g   = torch.tanh(g_raw);     o   = torch.sigmoid(o_raw)
        c   = f * c + i_g * g;       h   = o * torch.tanh(c)
        for key, val in zip(('f','i','g','o','h','c'), (f, i_g, g, o, h, c)):
            out[key].append(val.squeeze().cpu().numpy())

    return {k: np.stack(v) for k, v in out.items()}   # each (16, 256)


model.eval()
with torch.no_grad():
    states = collect_lstm_states(patchify(sample_img))

steps = range(num_patches)

fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"LSTM Internals  ·  Digit: {sample_label}", fontsize=12)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

gate_cfg = [
    ('f', "Forget gate (f)",         'crimson',    (0, 1)),
    ('i', "Input gate (i)",          'steelblue',  (0, 1)),
    ('g', "Cell candidate (g)",      'seagreen',   (-1, 1)),
    ('o', "Output gate (o)",         'darkorange', (0, 1)),
]
for col, (key, title, color, ylim) in enumerate(gate_cfg):
    ax   = fig.add_subplot(gs[0, col])
    mean = states[key].mean(axis=1)
    std  = states[key].std(axis=1)
    ax.plot(steps, mean, color=color, linewidth=2, marker='o', markersize=3)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(*ylim);  ax.set_xlim(-0.5, 15.5)
    ax.set_xticks(range(0, 16, 2))
    ax.set_xlabel("Patch step", fontsize=8)
    ax.grid(alpha=0.3)

ax_h = fig.add_subplot(gs[1, :])
im_h = ax_h.imshow(states['h'], aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
ax_h.set_title("Hidden state  h_t  —  short-term output (dao động nhiều hơn)", fontsize=9)
ax_h.set_ylabel("Patch step");  ax_h.set_xlabel("Neuron (0–255)")
plt.colorbar(im_h, ax=ax_h, fraction=0.015)

ax_c = fig.add_subplot(gs[2, :])
abs_max = float(np.abs(states['c']).max())
im_c = ax_c.imshow(states['c'], aspect='auto', cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
ax_c.set_title("Cell state  c_t  —  long-term memory (tích lũy chậm, giá trị lớn hơn)", fontsize=9)
ax_c.set_ylabel("Patch step");  ax_c.set_xlabel("Neuron (0–255)")
plt.colorbar(im_c, ax=ax_c, fraction=0.015)

plt.savefig('./output/lstm/lstm_internals.png', dpi=150)
print("💾 Saved: lstm_internals.png")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 4: Forget gate analysis — spatial + per digit class
#   Left  — 4×4 heatmap: vị trí không gian nào LSTM "quên" nhiều nhất
#            (f thấp = quên nhiều = reset memory, f cao = giữ lại)
#   Right — per-digit line chart: 10 chữ số có pattern forget khác nhau không?
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
forget_per_class = torch.zeros(10, num_patches)
forget_global    = torch.zeros(num_patches)
count_per_class  = torch.zeros(10)
n_total          = 0

with torch.no_grad():
    for data, labels in test_loader:
        data    = data.to(device)
        B       = data.size(0)
        patches = patchify(data)
        h = torch.zeros(B, hidden_size, device=device)
        c = torch.zeros(B, hidden_size, device=device)

        for t in range(num_patches):
            combined = torch.cat([patches[:, t], h], dim=1)
            f_raw, i_raw, g_raw, o_raw = model.lstm_cell.linear(combined).chunk(4, dim=1)
            f   = torch.sigmoid(f_raw);  i_g = torch.sigmoid(i_raw)
            g   = torch.tanh(g_raw);     o   = torch.sigmoid(o_raw)
            c   = f * c + i_g * g;       h   = o * torch.tanh(c)

            f_mean = f.mean(dim=1).cpu()   # (B,) mean over neurons
            forget_global[t] += f_mean.sum()
            for cls in range(10):
                mask = (labels == cls)
                if mask.any():
                    forget_per_class[cls, t] += f_mean[mask].sum()

        count_per_class += torch.tensor(
            [(labels == cls).sum().item() for cls in range(10)], dtype=torch.float
        )
        n_total += B

forget_global    /= n_total
forget_per_class /= count_per_class.unsqueeze(1)
forget_grid       = forget_global.numpy().reshape(grid_size, grid_size)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Forget Gate Analysis  ·  Mean f  (thấp = quên nhiều, cao = giữ lại)", fontsize=11)

im = axes[0].imshow(forget_grid, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
for r in range(grid_size):
    for c_idx in range(grid_size):
        axes[0].text(c_idx, r, f"{forget_grid[r, c_idx]:.3f}",
                     ha='center', va='center', fontsize=9, color='black')
axes[0].set_title("Mean forget gate per patch position")
axes[0].set_xlabel("Patch col");  axes[0].set_ylabel("Patch row")
plt.colorbar(im, ax=axes[0])

cmap10 = plt.cm.tab10
for cls in range(10):
    axes[1].plot(range(num_patches), forget_per_class[cls].numpy(),
                 label=str(cls), color=cmap10(cls / 10), linewidth=1.5, alpha=0.85)
axes[1].set_xlabel("Patch step (0=top-left → 15=bottom-right)")
axes[1].set_ylabel("Mean forget gate value")
axes[1].set_title("Forget gate per digit class")
axes[1].legend(title="Digit", ncol=2, fontsize=8)
axes[1].set_ylim(0, 1);  axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./output/lstm/forget_analysis.png', dpi=150)
print("💾 Saved: forget_analysis.png")

print("\n✅ All done!")
