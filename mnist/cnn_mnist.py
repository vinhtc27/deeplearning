import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os

# =====================
# Thiết bị tính toán
# =====================
# torch.device() chọn backend phần cứng để chạy tensor:
#   - "mps"  : Apple Metal Performance Shaders — GPU tích hợp trên chip M1/M2/M3
#   - "cuda" : GPU NVIDIA (không có ở đây nhưng có thể thêm nếu cần)
#   - "cpu"  : CPU thông thường — chậm hơn nhưng luôn khả dụng
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =====================
# Siêu tham số (Hyperparameters)
# =====================
batch_size = 64
epochs = 5
lr = 0.1

model_path = "./output/cnn/cnn_best.pt"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# =====================
# Dữ liệu (Data Pipeline)
# =====================
# LeNet-5 gốc (LeCun 1998) thiết kế cho ảnh 32×32
# MNIST chỉ có 28×28 → pad thêm 2 pixel mỗi cạnh để đúng kích thước gốc
# transforms.Compose: nối nhiều transform thành pipeline
transform = transforms.Compose([
    transforms.Pad(2),        # 28×28 → 32×32: thêm viền 0 (zero-padding) xung quanh
    transforms.ToTensor(),    # PIL → float32 tensor, chuẩn hóa về [0, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# =====================
# Kiến trúc Model — LeNet-5 (LeCun et al., 1998)
# =====================
# Kiến trúc gốc với một số điều chỉnh nhỏ:
#   - Giữ nguyên: filter size, số channel, pooling stride, FC size
#   - Thay Sigmoid/Tanh → ReLU (ổn định gradient hơn, không ảnh hưởng cấu trúc)
#   - Thay AvgPool → MaxPool (phổ biến hơn, giữ đặc trưng nổi bật hơn)
#   - Output: Linear(84→10) thay vì RBF units như bản gốc
#
# Luồng dữ liệu:
#   Input  : (B, 1, 32, 32)
#   C1     : Conv2d(1→6,   5×5) → (B,  6, 28, 28)
#   S2     : MaxPool(2×2)       → (B,  6, 14, 14)
#   C3     : Conv2d(6→16,  5×5) → (B, 16, 10, 10)
#   S4     : MaxPool(2×2)       → (B, 16,  5,  5)
#   C5     : Conv2d(16→120,5×5) → (B,120,  1,  1)  ← hoạt động như Linear (input vừa đúng 5×5)
#   Flatten:                      (B, 120)
#   F6     : Linear(120→84)     → (B,  84)
#   Output : Linear(84→10)      → (B,  10) logits
#
# Tổng tham số:
#   C1:     6×(1×5×5)  +   6 =     156
#   C3:    16×(6×5×5)  +  16 =   2.416
#   C5:   120×(16×5×5) + 120 =  48.120
#   F6:    84×120      +  84 =  10.164
#   Out:   10×84       +  10 =     850
#   Tổng: 61.706
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # C1 — phát hiện các đặc trưng cục bộ đơn giản (cạnh, góc) trên ảnh gốc
            nn.Conv2d(1, 6, kernel_size=5),     # (B,1,32,32) → (B,6,28,28)
            nn.ReLU(),
            # S2 — giảm kích thước không gian 2×, tăng tính bất biến với dịch chuyển nhỏ
            nn.MaxPool2d(kernel_size=2),         # (B,6,28,28) → (B,6,14,14)

            # C3 — kết hợp đặc trưng từ C1 để học pattern phức tạp hơn (cong, chéo)
            nn.Conv2d(6, 16, kernel_size=5),    # (B,6,14,14) → (B,16,10,10)
            nn.ReLU(),
            # S4
            nn.MaxPool2d(kernel_size=2),         # (B,16,10,10) → (B,16,5,5)

            # C5 — conv với kernel đúng bằng kích thước feature map → tương đương Linear
            # Giữ nguyên dạng Conv2d theo thiết kế gốc của LeCun
            nn.Conv2d(16, 120, kernel_size=5),  # (B,16,5,5) → (B,120,1,1)
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),           # (B,120,1,1) → (B,120)
            # F6 — lớp fully connected, học cách kết hợp 120 đặc trưng cao cấp
            nn.Linear(120, 84),
            nn.ReLU(),
            # Output — 10 logits, không activation (CrossEntropyLoss tích hợp Softmax)
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Khởi tạo model và chuyển tất cả tham số sang thiết bị đã chọn
model = LeNet5().to(device)

print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

# =====================
# Khởi tạo baseline accuracy
# =====================
# Luôn huấn luyện từ đầu với trọng số ngẫu nhiên — không load checkpoint cũ
# Mục đích: mỗi lần chạy là một thí nghiệm độc lập, kết quả phản ánh đúng
# hiệu quả của từng kĩ thuật tối ưu, không bị ảnh hưởng bởi các lần chạy trước
best_acc = 0.0

# =====================
# Hàm mất mát & Thuật toán tối ưu
# =====================
# nn.CrossEntropyLoss(): kết hợp LogSoftmax + NLLLoss, nhận logits thô
criterion = nn.CrossEntropyLoss()

# optim.SGD: W = W - lr * grad(W)
optimizer = optim.SGD(model.parameters(), lr=lr)

# =====================
# Vòng lặp huấn luyện (Training Loop)
# =====================
for epoch in range(epochs):
    start_time = time.time()

    # ---- GIAI ĐOẠN HUẤN LUYỆN (TRAIN PHASE) ----
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()          # Xóa gradient từ bước trước
        output = model(data)           # Forward pass: (B,1,32,32) qua toàn bộ LeNet-5
        loss = criterion(output, target)
        loss.backward()                # Backward pass: tính gradient qua conv + linear
        optimizer.step()               # Cập nhật trọng số

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch+1} | Batch {batch_idx}] Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # ---- GIAI ĐOẠN ĐÁNH GIÁ (EVAL PHASE) ----
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    epoch_time = time.time() - start_time

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Test Loss:  {avg_test_loss:.4f}")
    print(f"Accuracy:   {acc:.2f}%")
    print(f"Time:       {epoch_time:.2f}s\n")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), model_path)
        print(f"🔥 New best model saved! ({best_acc:.2f}%)\n")
    else:
        print(f"❌ No improvement (Best: {best_acc:.2f}%)\n")

print(f"✅ Training finished. Best Accuracy: {best_acc:.2f}%")

# =====================
# Biểu diễn trọng số và feature maps của các conv layer
# =====================

model.eval()
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images[:8].to(device)  # (8, 1, 32, 32)

# --- 1. FILTERS của C1 ---
# features[0] = Conv2d(1→6, 5×5): weight shape (6, 1, 5, 5)
# Chỉ có 6 filter — nhìn rõ từng pattern mà C1 học được
# Màu đỏ = trọng số dương (kích hoạt khi pixel sáng), xanh = âm (ức chế)
filters_c1 = model.features[0].weight.detach().cpu().numpy()  # (6, 1, 5, 5)

fig, axes = plt.subplots(1, 6, figsize=(12, 2))
fig.suptitle("C1 Filters — 6 filter 5×5 (đỏ = kích hoạt, xanh = ức chế)", fontsize=12)
for i, ax in enumerate(axes):
    f = filters_c1[i, 0]
    abs_max = abs(f).max()
    ax.imshow(f, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    ax.set_title(f"F{i}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('./output/cnn/c1_filters.png', dpi=150)
print("💾 Đã lưu: c1_filters.png")

# --- 2. FEATURE MAPS của C1 (sau S2) ---
# Chạy qua features[0:3] = Conv2d → ReLU → MaxPool → (8, 6, 14, 14)
# Mỗi ô = 1 feature map 14×14 — vùng sáng = nơi filter đó phản ứng mạnh
with torch.no_grad():
    fmaps_c1 = model.features[:3](sample_images).cpu().numpy()  # (8, 6, 14, 14)

fig, axes = plt.subplots(8, 6, figsize=(9, 12))
fig.suptitle("C1 Feature Maps (sau ReLU + MaxPool, 14×14)\n"
             "hàng = ảnh đầu vào, cột = filter", fontsize=12)
for img_idx in range(8):
    for f_idx in range(6):
        ax = axes[img_idx, f_idx]
        ax.imshow(fmaps_c1[img_idx, f_idx], cmap='viridis')
        ax.axis('off')
        if img_idx == 0:
            ax.set_title(f"F{f_idx}", fontsize=9)
    axes[img_idx, 0].set_ylabel(f"Label {sample_labels[img_idx].item()}", fontsize=9)
plt.tight_layout()
plt.savefig('./output/cnn/c1_feature_maps.png', dpi=150)
print("💾 Đã lưu: c1_feature_maps.png")

# --- 3. FILTERS của C3 ---
# features[3] = Conv2d(6→16, 5×5): weight shape (16, 6, 5, 5)
# Mỗi filter có 6 channel → lấy trung bình theo chiều input để xem "độ mạnh" tổng thể
filters_c3 = model.features[3].weight.detach().cpu().numpy()  # (16, 6, 5, 5)
filters_c3_mean = filters_c3.mean(axis=1)                     # (16, 5, 5)

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
fig.suptitle("C3 Filters — 16 filter 5×5, trung bình qua 6 input channels\n"
             "(đỏ = kích hoạt, xanh = ức chế)", fontsize=12)
for i, ax in enumerate(axes.flat):
    f = filters_c3_mean[i]
    abs_max = abs(f).max()
    ax.imshow(f, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    ax.set_title(f"F{i}", fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig('./output/cnn/c3_filters.png', dpi=150)
print("💾 Đã lưu: c3_filters.png")

# --- 4. FEATURE MAPS của C3 (sau S4) ---
# Chạy qua features[0:6] = C1→ReLU→S2→C3→ReLU→S4 → (8, 16, 5, 5)
# Feature map 5×5 rất nhỏ — chứa đặc trưng rất trừu tượng (hình dạng tổng thể của chữ số)
with torch.no_grad():
    fmaps_c3 = model.features[:6](sample_images).cpu().numpy()  # (8, 16, 5, 5)

fig, axes = plt.subplots(8, 16, figsize=(18, 9))
fig.suptitle("C3 Feature Maps (sau ReLU + MaxPool, 5×5)\n"
             "hàng = ảnh đầu vào, cột = filter — feature map nhỏ = đặc trưng trừu tượng hơn", fontsize=12)
for img_idx in range(8):
    for f_idx in range(16):
        ax = axes[img_idx, f_idx]
        ax.imshow(fmaps_c3[img_idx, f_idx], cmap='viridis')
        ax.axis('off')
        if img_idx == 0:
            ax.set_title(f"F{f_idx}", fontsize=8)
    axes[img_idx, 0].set_ylabel(f"Label {sample_labels[img_idx].item()}", fontsize=9)
plt.tight_layout()
plt.savefig('./output/cnn/c3_feature_maps.png', dpi=150)
print("💾 Đã lưu: c3_feature_maps.png")
