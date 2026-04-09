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
# is_available() kiểm tra xem MPS có sẵn trên máy không trước khi dùng
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =====================
# Siêu tham số (Hyperparameters)
# =====================
# Đây là các tham số cấu hình quá trình huấn luyện — không được học từ dữ liệu
# mà do người dùng thiết lập thủ công và tinh chỉnh (tuning)

batch_size = 64     # Số mẫu dữ liệu xử lý song song trong mỗi bước (mini-batch)
                    # Batch lớn → ổn định hơn nhưng tốn nhiều RAM; batch nhỏ → nhiễu hơn nhưng generalize tốt hơn

epochs = 5          # Số lần duyệt toàn bộ tập huấn luyện
                    # Mỗi epoch = model thấy tất cả 60.000 ảnh một lần

lr = 0.1            # Learning rate (tốc độ học) — điều chỉnh mức độ cập nhật trọng số sau mỗi bước
                    # lr quá lớn → dao động, không hội tụ; lr quá nhỏ → học rất chậm

model_path = "./output/mlp/mlp_best.pt"  # Tên file lưu trọng số model tốt nhất (PyTorch Tensor dictionary)
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

# =====================
# Dữ liệu (Data Pipeline)
# =====================
# transforms.ToTensor():
#   - Chuyển ảnh PIL (0–255, uint8) → torch.Tensor (0.0–1.0, float32)
#   - Reshape từ (H, W) → (C, H, W) — tức (1, 28, 28) với ảnh grayscale
#   - Chuẩn hóa về [0, 1] giúp gradient ổn định hơn trong quá trình lan truyền ngược
transform = transforms.ToTensor()

# datasets.MNIST() tải bộ dữ liệu MNIST:
#   - 60.000 ảnh chữ số viết tay (0–9) cho tập huấn luyện
#   - 10.000 ảnh cho tập kiểm tra
#   - Mỗi ảnh là grayscale 28×28 pixel
#   - root='./data': thư mục lưu dữ liệu sau khi tải về
#   - download=True: tự động tải nếu chưa có
#   - train=True/False: chọn tập train hoặc test
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)

# DataLoader bao bọc dataset và tạo iterator trả về từng batch:
#   - shuffle=True  (train): xáo trộn ngẫu nhiên để tránh model học theo thứ tự cố định
#   - shuffle=False (test) : giữ nguyên thứ tự để kết quả đánh giá nhất quán
#   - num_workers=0 mặc định: dùng 1 thread chính (có thể tăng để tải song song)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# =====================
# Kiến trúc Model
# =====================
# Mạng nơ-ron 3 lớp (2 hidden layers):
#   Input (784) → Linear → ReLU → Linear → ReLU → Linear → Output (10)
# nn.Sequential gom toàn bộ pipeline vào self.net — thêm/bớt layer chỉ cần sửa đây
# Tham số:
#   Linear(784→512): 784×512 + 512 = 401.920
#   Linear(512→512): 512×512 + 512 = 262.656
#   Linear(512→10) :  512×10  +  10 =   5.130
#   Tổng: 669.706
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),               # (B,1,28,28) → (B,784): trải phẳng ảnh thành vector 1 chiều
            nn.Linear(28*28, 512),      # Hidden layer 1: W(512×784) + b(512) — học đặc trưng từ pixel
            nn.ReLU(),                  # ReLU(x) = max(0,x): thêm phi tuyến, cắt giá trị âm về 0
            nn.Linear(512, 512),        # Hidden layer 2: W(512×512) + b(512) — học đặc trưng bậc cao hơn
            nn.ReLU(),                  # ReLU lần nữa để duy trì phi tuyến giữa các lớp ẩn
            nn.Linear(512, 10),         # Output layer: 10 logits cho 10 chữ số (0–9), không activation
                                        # CrossEntropyLoss tích hợp sẵn Softmax nên không cần thêm ở đây
        )

    def forward(self, x):
        return self.net(x)

# Khởi tạo model và chuyển tất cả tham số sang thiết bị đã chọn (MPS/CPU)
# .to(device) đảm bảo tensor của model và tensor dữ liệu ở cùng thiết bị
model = MLP().to(device)

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
# nn.CrossEntropyLoss():
#   - Kết hợp LogSoftmax + NLLLoss thành một bước duy nhất
#   - Nhận logits (chưa qua softmax) làm đầu vào → không cần thêm softmax trong model
#   - Công thức: L = -log(softmax(output)[true_class])
#   - Phù hợp cho bài toán phân loại nhiều lớp (multiclass classification)
criterion = nn.CrossEntropyLoss()

# optim.SGD (Stochastic Gradient Descent):
#   - Thuật toán tối ưu cơ bản nhất
#   - Cập nhật trọng số: W = W - lr * grad(W)
#   - model.parameters(): truyền tất cả tham số cần học (W, b của lớp Linear)
#   - Có thể thêm momentum=0.9, weight_decay=1e-4 để cải thiện hội tụ
optimizer = optim.SGD(model.parameters(), lr=lr)

# =====================
# Vòng lặp huấn luyện (Training Loop)
# =====================
for epoch in range(epochs):
    start_time = time.time()  # Ghi nhận thời điểm bắt đầu để tính thời gian mỗi epoch

    # ---- GIAI ĐOẠN HUẤN LUYỆN (TRAIN PHASE) ----
    model.train()   # Chuyển model sang chế độ train: bật Dropout (nếu có), BatchNorm dùng batch stats
    train_loss = 0.0  # Tích lũy tổng loss để tính trung bình sau epoch

    # Duyệt qua từng mini-batch trong tập huấn luyện
    # batch_idx: chỉ số batch (0, 1, 2, ...)
    # data: tensor ảnh (batch_size, 1, 28, 28)
    # target: tensor nhãn (batch_size,) — giá trị nguyên từ 0–9
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Chuyển batch lên thiết bị tính toán

        # Bước 1: Xóa gradient từ bước trước
        # PyTorch tích lũy gradient theo mặc định — phải reset về 0 trước mỗi backward pass
        optimizer.zero_grad()

        # Bước 2: Forward pass — đưa dữ liệu qua model để tính logits
        output = model(data)  # (batch_size, 10)

        # Bước 3: Tính loss — so sánh logits với nhãn thật
        loss = criterion(output, target)  # Scalar tensor

        # Bước 4: Backward pass — tính gradient của loss theo tất cả tham số
        # PyTorch tự động dùng chain rule (autograd) để lan truyền ngược
        loss.backward()

        # Bước 5: Cập nhật trọng số dựa theo gradient vừa tính
        # W = W - lr * grad(W) với SGD cơ bản
        optimizer.step()

        # .item() chuyển scalar tensor → float Python để tính toán nhanh hơn
        train_loss += loss.item()

        # In loss mỗi 100 batch để theo dõi quá trình hội tụ trong epoch
        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch+1} | Batch {batch_idx}] Loss: {loss.item():.4f}")

    # Tính loss trung bình trên toàn tập train trong epoch này
    avg_train_loss = train_loss / len(train_loader)

    # ---- GIAI ĐOẠN ĐÁNH GIÁ (EVAL PHASE) ----
    model.eval()   # Chuyển sang chế độ đánh giá
    correct = 0
    total = 0
    test_loss = 0.0

    # Không tính gradient trong quá trình đánh giá — chỉ cần forward pass
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)                  # Forward pass
            loss = criterion(output, target)      # Tính test loss
            test_loss += loss.item()

            # Lấy lớp có điểm số logit cao nhất làm dự đoán
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    # Tính accuracy: tỉ lệ phần trăm dự đoán đúng trên toàn tập test (10.000 mẫu)
    acc = 100 * correct / total

    # Tính test loss trung bình
    avg_test_loss = test_loss / len(test_loader)

    # Tổng thời gian chạy epoch (train + eval)
    epoch_time = time.time() - start_time

    # ---- IN KẾT QUẢ EPOCH ----
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}")   # Loss trên tập train (càng nhỏ càng tốt)
    print(f"Test Loss:  {avg_test_loss:.4f}")    # Loss trên tập test (theo dõi overfitting)
    print(f"Accuracy:   {acc:.2f}%")             # Độ chính xác trên tập test
    print(f"Time:       {epoch_time:.2f}s\n")    # Thời gian chạy epoch

    # ---- LƯU MODEL NẾU TỐT HƠN TRƯỚC ----
    # Chỉ lưu khi accuracy cải thiện so với lần tốt nhất trước đó
    # → Đảm bảo file best.pt luôn chứa model có hiệu năng cao nhất
    if acc > best_acc:
        best_acc = acc
        # torch.save(state_dict, path): chỉ lưu trọng số (không lưu cấu trúc model)
        # state_dict là OrderedDict: { "fc.weight": tensor, "fc.bias": tensor }
        torch.save(model.state_dict(), model_path)
        print(f"🔥 New best model saved! ({best_acc:.2f}%)\n")
    else:
        # Model không cải thiện — không ghi đè file để giữ lại model tốt nhất
        print(f"❌ No improvement (Best: {best_acc:.2f}%)\n")

# =====================
# Kết thúc huấn luyện
# =====================
# In tóm tắt kết quả sau tất cả các epoch
print(f"✅ Training finished. Best Accuracy: {best_acc:.2f}%")

# =====================
# Biểu diễn trọng số và kích hoạt của các lớp ẩn
# =====================

# Lấy 8 ảnh mẫu từ test_loader để dùng chung cho phần activation
model.eval()
sample_images, sample_labels = next(iter(test_loader))  # Lấy batch đầu tiên
sample_images = sample_images[:8].to(device)            # Chỉ dùng 8 ảnh đầu

# --- 1. WEIGHTS của hidden layer 1 ---
# net[1] = Linear(784→512): shape (512, 784)
# Reshape mỗi hàng → (28, 28) để thấy vùng ảnh mà neuron đó phản ứng mạnh
# Màu đỏ = trọng số dương (pixel sáng → neuron kích hoạt mạnh)
# Màu xanh = trọng số âm (pixel sáng → neuron bị ức chế)
weights_h1 = model.net[1].weight.detach().cpu().numpy()  # (512, 784)

fig, axes = plt.subplots(16, 32, figsize=(32, 16))
fig.suptitle("Hidden Layer 1 Weights — mỗi ô = 1 neuron nhìn vào ảnh 28×28\n"
             "(đỏ = kích hoạt, xanh = ức chế)", fontsize=13)
for i, ax in enumerate(axes.flat):
    w = weights_h1[i].reshape(28, 28)
    abs_max = abs(w).max()                              # Chuẩn hóa colormap quanh 0
    ax.imshow(w, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    ax.axis('off')
plt.tight_layout()
plt.savefig('./output/mlp/hidden1_weights.png', dpi=150)
print("💾 Đã lưu: mlp_hidden1_weights.png")

# --- 2. ACTIVATIONS của hidden layer 1 với 8 ảnh mẫu ---
# Chạy forward qua net[0:3] = Flatten → Linear(784→512) → ReLU
# → thấy 512 neuron kích hoạt bao nhiêu với từng ảnh đầu vào cụ thể
with torch.no_grad():
    act_h1 = model.net[:3](sample_images).cpu().numpy()   # (8, 512)

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
fig.suptitle("Activations của Hidden Layer 1 (sau ReLU) — mỗi cột = 1 neuron\n"
             "Giá trị 0 = neuron bị tắt (ReLU cắt âm)", fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.bar(range(512), act_h1[i], width=1.0, color='steelblue')
    ax.set_title(f"Label: {sample_labels[i].item()}", fontsize=10)
    ax.set_xlabel("Neuron index (0–511)")
    ax.set_ylabel("Activation")
    ax.set_xlim(0, 512)
plt.tight_layout()
plt.savefig('./output/mlp/hidden1_activations.png', dpi=150)
print("💾 Đã lưu: mlp_hidden1_activations.png")

# --- 3. WEIGHTS của hidden layer 2 ---
# net[3] = Linear(512→512): shape (512, 512)
# Không thể reshape thành ảnh 28×28 vì input là 512 (không phải pixel)
# → Dùng imshow trực tiếp để thấy ma trận kết nối giữa 512 neuron lớp 1 và 512 neuron lớp 2
weights_h2 = model.net[3].weight.detach().cpu().numpy()  # (512, 512)

fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle("Hidden Layer 2 Weights — ma trận kết nối (512×512)\n"
             "(đỏ = kết nối dương, xanh = kết nối âm)", fontsize=13)
abs_max = abs(weights_h2).max()
ax.imshow(weights_h2, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, aspect='auto')
ax.set_xlabel("Neuron layer 1 (0–511)")
ax.set_ylabel("Neuron layer 2 (0–511)")
plt.tight_layout()
plt.savefig('./output/mlp/hidden2_weights.png', dpi=150)
print("💾 Đã lưu: mlp_hidden2_weights.png")

# --- 4. ACTIVATIONS của hidden layer 2 với 8 ảnh mẫu ---
# Chạy forward qua net[0:5] = Flatten → Linear → ReLU → Linear(512→512) → ReLU
# → thấy đặc trưng bậc cao hơn sau khi qua 2 lớp ẩn
with torch.no_grad():
    act_h2 = model.net[:5](sample_images).cpu().numpy()   # (8, 512)

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
fig.suptitle("Activations của Hidden Layer 2 (sau ReLU) — mỗi cột = 1 neuron\n"
             "Giá trị 0 = neuron bị tắt (ReLU cắt âm)", fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.bar(range(512), act_h2[i], width=1.0, color='darkorange')
    ax.set_title(f"Label: {sample_labels[i].item()}", fontsize=10)
    ax.set_xlabel("Neuron index (0–511)")
    ax.set_ylabel("Activation")
    ax.set_xlim(0, 512)
plt.tight_layout()
plt.savefig('./output/mlp/hidden2_activations.png', dpi=150)
print("💾 Đã lưu: mlp_hidden2_activations.png")
