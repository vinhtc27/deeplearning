import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report
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


# Cải tiến 4.1: Giảm batch size từ 100 xuống 64 để phù hợp với GPU MPS có bộ nhớ hạn chế
batch_size = 64     # Số mẫu dữ liệu xử lý song song trong mỗi bước (mini-batch)
                    # Batch lớn → ổn định hơn nhưng tốn nhiều RAM; batch nhỏ → nhiễu hơn nhưng generalize tốt hơn

# Cải tiến 4.1: Tăng số epoch lên 100 để model có đủ thời gian hội tụ với các kĩ thuật tối ưu đã thêm
epochs = 100        # Số lần duyệt toàn bộ tập huấn luyện
                    # Mỗi epoch = model thấy tất cả 50.000 ảnh một lần

lr = 0.1            # Learning rate (tốc độ học) — điều chỉnh mức độ cập nhật trọng số sau mỗi bước
                    # lr quá lớn → dao động, không hội tụ; lr quá nhỏ → học rất chậm

model_path = "./output/mlp/mlp_best.pt"  # Tên file lưu trọng số model tốt nhất (PyTorch Tensor dictionary)
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

# =====================
# Dữ liệu (Data Pipeline)
# =====================
# transforms.ToTensor():
#   - Chuyển ảnh PIL (0–255, uint8) → torch.Tensor (0.0–1.0, float32)
#   - Reshape từ (H, W) → (C, H, W) — tức (3, 32, 32) với ảnh RGB
#   - Chuẩn hóa về [0, 1] giúp gradient ổn định hơn trong quá trình lan truyền ngược
transform = transforms.ToTensor()

# Các cải tiến dựa trên tài liệu
# Baseline: https://cs231n.github.io/neural-networks-1
# -> Accuracy 40–43% sau 5 epoch
# Cải tiến 2: https://cs231n.github.io/neural-networks-2
# -> Accuracy tăng lên 45% sau 5 epoch, 50% sau 20 epoch, 55% sau 50 epoch
# Cải tiến 3: https://cs231n.github.io/neural-networks-3
# -> Accuracy tăng lên 50% sau 5 epoch, 56% sau 20 epoch, 59% sau 50 epoch
# Cải tiến 4: Sau khi có kết quả đánh giá từ các kĩ thuật ở cải tiến 3
# -> Accuracy hội tụ chậm hơn:
#   + giảm overfitting cho phép nhiều epochs hơn (augmentation + dropout + max norm constraint)
#   + giúp model học đặc trưng tốt hơn (theo trực quan hóa hidden layer weights và activations)
#   + tăng generalization trên dữ liệu unseen (theo trực quan hóa confusion matrix)
#   + giảm test loss và tăng accuracy trên tập test (theo curve loss và accuracy)
#   + giảm dead neuron sau khi đánh giá (theo trực quan hóa activation)
# Cải tiến 5: Bổ sung visualization theo chuẩn nghiên cứu hơn
# Cải tiến 6: Sau khi có kết quả đánh giá trực quan từ cải tiến 5
# -> Các thay đổi dựa trên phân tích output/mlp:
#   + 6.1: Giảm Dropout H1 từ 0.3 → 0.2 (train acc 52% < test 58% → regularization quá mạnh)
#   + 6.2: Thu hẹp H2 từ 512→256 (H2 weights gần 0 toàn bộ, 512→512 đang redundant)
#   + 6.3: Tăng LeakyReLU negative_slope H2 từ 0.2 → 0.3 (H2 dead neurons ~35%)
#   + 6.4: Giảm Dropout H2 từ 0.2 → 0.1 (H2 đang bị triệt tiêu quá mức)
#   + 6.5: Thêm gradient clipping max_norm=1.0 (gradient norm tăng liên tục 0.6→1.4)
#   + 6.6: Đổi CosineAnnealingLR → OneCycleLR (loss giảm chậm ở giữa, step theo batch)

# Cải tiến 2.1: Chuẩn hóa theo mean/std của CIFAR10 (không chia std để giữ nguyên độ tương phản)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
#                          std=(1,1,1))
# ])

# Cải tiến 2.2: Chuẩn hóa theo mean/std của CIFAR10 (có chia std để giảm độ tương phản, giúp hội tụ nhanh hơn)
train_transform = transforms.Compose([
    # Cải tiến 4.2: Thêm Augmentation (RandomHorizontalFlip + RandomCrop) để tăng đa dạng dữ liệu, giảm overfitting
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    # Cải tiến 4.2: Không áp dụng augmentation cho tập test để đánh giá chính xác hiệu năng trên dữ liệu thật
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

# datasets.CIFAR10() tải bộ dữ liệu CIFAR10:
#   - 50.000 ảnh cho tập huấn luyện
#   - 10.000 ảnh cho tập kiểm tra
#   - Mỗi ảnh là RGB 32×32 pixel
#   - root='./data': thư mục lưu dữ liệu sau khi tải về
#   - download=True: tự động tải nếu chưa có
#   - train=True/False: chọn tập train hoặc test
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=test_transform)
class_names = test_dataset.classes

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
#   Input (3072) → Linear → ReLU → Linear → ReLU → Linear → Output (10)
# nn.Sequential gom toàn bộ pipeline vào self.net — thêm/bớt layer chỉ cần sửa đây
# Tham số:
#   Linear(3072→512): 3072×512 + 512 = 1.573.376
#   # Cải tiến 6.2: Linear(512→256): 512×256 + 256 = 131.328
#   Linear(512→512): 512×512 + 512 = 262.656
#   # Cải tiến 6.2: Linear(256→10) :  256×10  +  10 =   2.570
#   Linear(512→10) :  512×10  +  10 =   5.130
#   Tổng: 1.841.162 → ~1.707.274 (với cải tiến 6.2)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),               # (B,3,32,32) → (B,3072): trải phẳng ảnh thành vector 1 chiều

			# Hidden layer 1:
            nn.Linear(3*32*32, 512),    # Hidden layer 1: W(512×3072) + b(512) — học đặc trưng từ pixel
			# Cải tiến 2.3: Batch Normalization sau Linear, trước ReLU (comment vì thay bằng 4.3)
			# nn.BatchNorm1d(512), 		# Giúp ổn định và tăng tốc hội tụ, giảm phụ thuộc vào khởi tạo
            # nn.ReLU(),                # ReLU(x) = max(0,x): thêm phi tuyến, cắt giá trị âm về 0
            # Cải tiến 4.3: Thay ReLU bằng LeakyReLU để giảm nguy cơ Dead Neuron sau khi đánh giá
            nn.LeakyReLU(negative_slope=0.2),
			# Cải tiến 4.4: Di chuyển BN sau ReLU để LeakyReLU hoạt động trên activation "thô sơ" trước khi BN chuẩn hóa
			nn.BatchNorm1d(512), 		# Giúp ổn định và tăng tốc hội tụ, giảm phụ thuộc vào khởi tạo
			# Cải tiến 2.4: Dropout sau ReLU để giảm overfitting (tắt ngẫu nhiên 50% neuron mỗi lần train)
			# nn.Dropout(p=0.5),          # Chỉ bật Dropout trong giai đoạn train, tự động tắt trong eval
            # Cải tiến 4.5: Giảm tỉ lệ Dropout xuống 0.3 để giữ lại nhiều neuron hơn sau khi đã chuyển sang LeakyReLU
            # nn.Dropout(p=0.3),
            # Cải tiến 6.1: Giảm thêm xuống 0.2 — train acc (52%) thấp hơn test (58%) cho thấy
            #               regularization H1 đang quá mạnh, model bị handicap khi train
            nn.Dropout(p=0.2),

			# Hidden layer 2:
            # Cải tiến 6.2: Thu hẹp H2 từ 512→256 — H2 weights gần 0 toàn bộ (visualize như ma trận trắng),
            #               512→512 đang redundant; compact hơn để force H2 học đặc trưng rõ ràng hơn
            # nn.Linear(512, 512),        # Hidden layer 2: W(512×512) + b(512) — học đặc trưng bậc cao hơn
            nn.Linear(512, 256),          # Hidden layer 2: W(256×512) + b(256)
			# Cải tiến 2.3: (comment vì thay bằng 4.3)
			# nn.BatchNorm1d(512),
            # nn.ReLU(),                  # ReLU lần nữa để duy trì phi tuyến giữa các lớp ẩn
            # Cải tiến 4.3: Thay ReLU bằng LeakyReLU để giảm nguy cơ Dead Neuron sau khi đánh giá
            # Cải tiến 6.3: Tăng negative_slope H2 từ 0.2 → 0.3 — H2 có ~35% dead neurons,
            #               cần gradient chảy qua vùng âm mạnh hơn để hồi sinh neurons
            # nn.LeakyReLU(negative_slope=0.2),
            nn.LeakyReLU(negative_slope=0.3),
			# Cải tiến 4.4: Di chuyển BN sau ReLU
            # Cải tiến 6.2: Cập nhật BN theo H2 mới (512→256)
			# nn.BatchNorm1d(512),
			nn.BatchNorm1d(256),
			# Cải tiến 2.4:
			# nn.Dropout(p=0.3),
            # Cải tiến 4.5: Giảm tỉ lệ Dropout xuống 0.2 ở lớp ẩn thứ 2 vì đã có LeakyReLU giảm nguy cơ dead neuron
            # Cải tiến 6.4: Giảm thêm xuống 0.1 — H2 đang bị triệt tiêu quá mức (dead neurons cao + weights gần 0)
            # nn.Dropout(p=0.2),
            nn.Dropout(p=0.1),

            # Cải tiến 6.2: Cập nhật output layer từ 512→10 sang 256→10 theo H2 mới
            # nn.Linear(512, 10),
            nn.Linear(256, 10),         # Output layer: 10 logits cho 10 lớp (0–9), không activation
                                        # CrossEntropyLoss tích hợp sẵn Softmax nên không cần thêm ở đây
        )

        # Cải tiến 2.5: He Initialization cho ReLU
        # sqrt(2/n) thay vì 1/sqrt(n) vì ReLU cắt ~50% activation → cần bù factor 2
        for m in self.net:
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                # Cải tiến 4.3: Thay nonlinearity thành 'leaky_relu' để phù hợp với activation function mới
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

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

# Cải tiến 3.1: Theo dõi metric toàn bộ quá trình train để debug và vẽ curve
train_loss_history = []
test_loss_history = []
acc_history = []
train_acc_history = []   # Cải tiến 5.1: theo dõi train acc để so sánh train/test (chuẩn nghiên cứu)
lr_history = []          # Cải tiến 5.2: theo dõi learning rate schedule theo từng epoch
grad_norm_history = []
dead_relu_h1_history = []
dead_relu_h2_history = []

# =====================
# Hàm mất mát & Thuật toán tối ưu
# =====================
# nn.CrossEntropyLoss():
#   - Kết hợp LogSoftmax + NLLLoss thành một bước duy nhất
#   - Nhận logits (chưa qua softmax) làm đầu vào → không cần thêm softmax trong model
#   - Công thức: L = -log(softmax(output)[true_class])
#   - Phù hợp cho bài toán phân loại nhiều lớp (multiclass classification)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# optim.SGD (Stochastic Gradient Descent):
#   - Thuật toán tối ưu cơ bản nhất
#   - Cập nhật trọng số: W = W - lr * grad(W)
#   - model.parameters(): truyền tất cả tham số cần học (W, b của lớp Linear)
#   - Có thể thêm momentum=0.9, weight_decay=1e-4 để cải thiện hội tụ
# optimizer = optim.SGD(model.parameters(), lr=0.01,
# # Cải tiến 2.6: L2 regularization (weight decay) để giảm overfitting
#                       weight_decay=1e-4) # đây chính là L2

# Cải tiến 3.2: Thêm Momentum cho SGD để cập nhật ổn định hơn
optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9,
                      weight_decay=1e-4)

# Cải tiến 3.3: Learning Rate Scheduler (CosineAnnealingLR) để giảm lr mượt theo epoch
# Cải tiến 6.6: Đổi sang OneCycleLR — loss giảm chậm ở giữa do CosineAnnealing không có warmup
#   OneCycleLR: lr thấp → tăng lên đỉnh (max_lr) → giảm cosine xuống rất thấp ("super-convergence")
#   Step theo batch (không phải epoch) → mượt hơn, thoát plateau nhanh hơn
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr,
    epochs=epochs, steps_per_epoch=len(train_loader)
)

# Cải tiến 3.4: Sanity check phân phối output ban đầu (softmax mean ~ 0.1 với 10 lớp)
model.eval()
with torch.no_grad():
    sanity_data, _ = next(iter(train_loader))
    sanity_data = sanity_data.to(device)
    sanity_output = model(sanity_data)
    sanity_probs = torch.softmax(sanity_output, dim=1)
    sanity_mean_probs = sanity_probs.mean(dim=0)
print(f"Initial softmax mean (10 lớp): {sanity_mean_probs.cpu().numpy()}")
print(f"Mean toàn bộ lớp: {sanity_mean_probs.mean().item():.4f} (kỳ vọng xấp xỉ 0.1)")

# Cải tiến 3.5: Overfit small batch sanity check (train trên 1 batch, kỳ vọng loss gần 0)
overfit_data, overfit_target = next(iter(train_loader))
overfit_data, overfit_target = overfit_data.to(device), overfit_target.to(device)
overfit_model = MLP().to(device)
overfit_model.load_state_dict(model.state_dict())
overfit_optimizer = optim.SGD(overfit_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
overfit_steps = 100
overfit_model.train()
for step in range(overfit_steps):
    overfit_optimizer.zero_grad()
    overfit_output = overfit_model(overfit_data)
    overfit_loss = nn.CrossEntropyLoss()(overfit_output, overfit_target)
    overfit_loss.backward()
    overfit_optimizer.step()
    if (step + 1) % 20 == 0:
        print(f"Overfit step {step+1}/{overfit_steps} - Loss: {overfit_loss.item():.4f}")
print(f"Final overfit loss (1 batch): {overfit_loss.item():.4f} (kỳ vọng gần 0)")
del overfit_model

# =====================
# Vòng lặp huấn luyện (Training Loop)
# =====================
for epoch in range(epochs):
    start_time = time.time()  # Ghi nhận thời điểm bắt đầu để tính thời gian mỗi epoch

    # ---- GIAI ĐOẠN HUẤN LUYỆN (TRAIN PHASE) ----
    model.train()   # Chuyển model sang chế độ train: bật Dropout (nếu có), BatchNorm dùng batch stats
    train_loss = 0.0  # Tích lũy tổng loss để tính trung bình sau epoch
    epoch_grad_norm = 0.0  # Cải tiến 3.6: Tích lũy gradient norm để theo dõi độ ổn định
    train_correct = 0   # Cải tiến 5.3: đếm đúng để tính train accuracy
    train_total = 0

    # Duyệt qua từng mini-batch trong tập huấn luyện
    # batch_idx: chỉ số batch (0, 1, 2, ...)
    # data: tensor ảnh (batch_size, 3, 32, 32)
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

        # Cải tiến 3.6: Gradient norm logging theo từng batch để phát hiện exploding/vanishing gradient
        grad_sq_sum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq_sum += p.grad.detach().data.norm(2).item() ** 2
        batch_grad_norm = grad_sq_sum ** 0.5
        epoch_grad_norm += batch_grad_norm

        # Cải tiến 6.5: Gradient clipping để kiểm soát gradient norm tăng liên tục (0.6→1.4 sau 100 epoch)
        # clip_grad_norm_ scale toàn bộ gradient xuống nếu L2 norm vượt max_norm=2.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        # Bước 5: Cập nhật trọng số dựa theo gradient vừa tính
        # W = W - lr * grad(W) với SGD cơ bản
        optimizer.step()

        # Cải tiến 6.6: OneCycleLR step theo batch (không phải epoch)
        scheduler.step()

		# Cải tiến 2.7: Max Norm Constraint
        for m in model.net:
            if isinstance(m, nn.Linear):
                m.weight.data = torch.renorm(
                    m.weight.data, p=2, dim=0, maxnorm=4.0
                )

        # .item() chuyển scalar tensor → float Python để tính toán nhanh hơn
        train_loss += loss.item()
        _, train_pred = torch.max(output, 1)
        train_correct += (train_pred == target).sum().item()
        train_total += target.size(0)

        # In loss mỗi 100 batch để theo dõi quá trình hội tụ trong epoch
        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch+1} | Batch {batch_idx}] Loss: {loss.item():.4f}")
            print(f"[Epoch {epoch+1} | Batch {batch_idx}] Grad Norm: {batch_grad_norm:.4f}")

    # Tính loss trung bình trên toàn tập train trong epoch này
    avg_train_loss = train_loss / len(train_loader)
    avg_grad_norm = epoch_grad_norm / len(train_loader)
    train_acc = 100 * train_correct / train_total  # Cải tiến 5.3: train accuracy để so sánh train/test gap

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
            # all_targets / all_predictions chỉ cần sau khi train xong (cho confusion matrix)
            # — không collect mỗi epoch để tránh waste

    # Tính accuracy: tỉ lệ phần trăm dự đoán đúng trên toàn tập test (10.000 mẫu)
    acc = 100 * correct / total

    # Tính test loss trung bình
    avg_test_loss = test_loss / len(test_loader)

    # Cải tiến 3.7: Dead ReLU check (tỉ lệ activation gần như bằng 0 ở hidden layer 1 và 2)
    # Với LeakyReLU, activation không bao giờ == 0, nên check |activation| < threshold
    with torch.no_grad():
        dead_data, _ = next(iter(test_loader))
        dead_data = dead_data.to(device)

        h1 = model.net[:4](dead_data)      # Flatten -> Linear -> BN -> ReLU
        h2 = model.net[4:8](h1)            # Dropout -> Linear -> BN -> ReLU (Dropout tắt ở eval)

        dead_relu_h1 = (torch.abs(h1) < 0.01).float().mean().item() * 100.0
        dead_relu_h2 = (torch.abs(h2) < 0.01).float().mean().item() * 100.0

    # Tổng thời gian chạy epoch (train + eval)
    epoch_time = time.time() - start_time

    # Cải tiến 3.1: Lưu metric theo từng epoch để vẽ biểu đồ
    train_loss_history.append(avg_train_loss)
    test_loss_history.append(avg_test_loss)
    acc_history.append(acc)
    train_acc_history.append(train_acc)  # Cải tiến 5.3: theo dõi train accuracy
    grad_norm_history.append(avg_grad_norm)
    dead_relu_h1_history.append(dead_relu_h1)
    dead_relu_h2_history.append(dead_relu_h2)

    # ---- IN KẾT QUẢ EPOCH ----
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}")   # Loss trên tập train (càng nhỏ càng tốt)
    print(f"Test Loss:  {avg_test_loss:.4f}")    # Loss trên tập test (theo dõi overfitting)
    print(f"Accuracy:   {acc:.2f}%")             # Độ chính xác trên tập test
    print(f"Grad Norm:  {avg_grad_norm:.4f}")    # Cải tiến 3.6: norm gradient trung bình mỗi epoch
    print(f"Dead ReLU H1: {dead_relu_h1:.2f}%")  # Cải tiến 3.7: tỉ lệ neuron bị tắt ở hidden 1
    print(f"Dead ReLU H2: {dead_relu_h2:.2f}%")  # Cải tiến 3.7: tỉ lệ neuron bị tắt ở hidden 2
    print(f"Time:       {epoch_time:.2f}s\n")    # Thời gian chạy epoch

    # Cải tiến 3.3: Cập nhật learning rate sau mỗi epoch
    # Cải tiến 6.6: OneCycleLR đã step theo batch bên trong loop — không step lại ở đây
    # scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)  # Cải tiến 5.2: lưu lr để vẽ schedule curve
    print(f"Learning Rate (after scheduler): {current_lr:.6f}\n")

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

# Cải tiến 3.8: Vẽ loss curve và accuracy curve
# Cải tiến 5.4: 1×3 layout chuẩn nghiên cứu — Loss | Train vs Test Accuracy | LR Schedule
# + đánh dấu best epoch, hiển thị train/test gap để đánh giá overfitting
epochs_axis = range(1, epochs + 1)
best_epoch = int(np.argmax(acc_history)) + 1  # epoch có test accuracy cao nhất

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'MLP Training on CIFAR-10  —  Best Test Acc: {best_acc:.2f}% @ Epoch {best_epoch}',
             fontsize=13, fontweight='bold')

# Subplot 1: Loss curve
axes[0].plot(epochs_axis, train_loss_history, label='Train Loss', color='royalblue', linewidth=2)
axes[0].plot(epochs_axis, test_loss_history,  label='Test Loss',  color='crimson',   linewidth=2)
axes[0].axvline(best_epoch, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'Best epoch ({best_epoch})')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Subplot 2: Accuracy curve — train vs test để quan sát overfitting gap (chuẩn paper)
axes[1].plot(epochs_axis, train_acc_history, label='Train Acc', color='royalblue', linewidth=2)
axes[1].plot(epochs_axis, acc_history,       label='Test Acc',  color='seagreen',  linewidth=2)
axes[1].axvline(best_epoch, color='gray', linestyle='--', linewidth=1, alpha=0.7)
axes[1].scatter([best_epoch], [acc_history[best_epoch - 1]],
                color='seagreen', s=80, zorder=5, label=f'Best: {best_acc:.2f}%')
axes[1].set_title('Accuracy Curve (Train vs Test)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# Subplot 3: Learning Rate Schedule — thể hiện cosine annealing (chuẩn nghiên cứu)
axes[2].plot(epochs_axis, lr_history, color='darkorchid', linewidth=2)
axes[2].set_title('Learning Rate Schedule (Cosine Annealing)')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./output/mlp/training_curves.png', dpi=150)
print("💾 Đã lưu: training_curves.png")

# Cải tiến 3.9: Vẽ debug curves cho gradient norm và dead ReLU ratio
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_axis, grad_norm_history, color='darkorange', linewidth=2)
axes[0].set_title('Gradient Norm Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('L2 Norm')
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_axis, dead_relu_h1_history, label='Dead ReLU H1', color='purple', linewidth=2)
axes[1].plot(epochs_axis, dead_relu_h2_history, label='Dead ReLU H2', color='brown', linewidth=2)
axes[1].set_title('Dead ReLU Ratio (%)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Percentage (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./output/mlp/debug_curves.png', dpi=150)
print("💾 Đã lưu: debug_curves.png")

# =====================
# Biểu diễn trọng số và kích hoạt của các lớp ẩn
# =====================

# Lấy 8 ảnh mẫu từ test_loader để dùng chung cho phần activation
model.eval()
sample_images, sample_labels = next(iter(test_loader))  # Lấy batch đầu tiên
sample_images = sample_images[:8].to(device)            # Chỉ dùng 8 ảnh đầu

# --- 1. WEIGHTS của hidden layer 1 ---
# net[1] = Linear(3072→512): shape (512, 3072)
# Reshape mỗi hàng → (3, 32, 32) để thấy vùng ảnh mà neuron đó phản ứng mạnh
# Màu đỏ = trọng số dương (pixel sáng → neuron kích hoạt mạnh)
# Màu xanh = trọng số âm (pixel sáng → neuron bị ức chế)
# Cải tiến 5.4: dùng make_grid (torchvision) thay vì 512 subplot thủ công — chuẩn CS231n / fast.ai
# scale_each=True: chuẩn hóa mỗi neuron độc lập → dễ thấy pattern từng neuron
weights_h1 = model.net[1].weight.detach().cpu()         # (512, 3072) tensor
w_grid = weights_h1.reshape(512, 3, 32, 32)             # (512, 3, 32, 32)
grid_img = make_grid(w_grid, nrow=32, padding=1, normalize=True, scale_each=True)  # (3, H, W)

fig, ax = plt.subplots(figsize=(32, 16))
fig.suptitle("Hidden Layer 1 Weights — mỗi ô = 1 neuron nhìn vào ảnh 32×32\n"
             "(sáng/tối = kích hoạt/ức chế; normalize per-neuron)", fontsize=13)
ax.imshow(grid_img.permute(1, 2, 0).numpy())
ax.axis('off')
plt.tight_layout()
plt.savefig('./output/mlp/hidden1_weights.png', dpi=150)
print("💾 Đã lưu: mlp_hidden1_weights.png")

# --- 2. ACTIVATIONS của hidden layer 1 với 8 ảnh mẫu ---
# Chạy forward qua net[0:3] = Flatten → Linear(3072→512) → LeakyReLU
# → thấy 512 neuron kích hoạt bao nhiêu với từng ảnh đầu vào cụ thể
# Cải tiến 5.5: histogram phân phối activation thay vì bar chart theo neuron index
#   → dễ thấy dead neuron (spike ở 0), phân phối dày đặc, kurtosis
#   → chuẩn nghiên cứu (xem Glorot 2010, He 2015, BatchNorm paper)
with torch.no_grad():
    act_h1 = model.net[:3](sample_images).cpu().numpy()   # (8, 512)

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
fig.suptitle("Activation Distribution — Hidden Layer 1 (sau LeakyReLU)\n"
             "spike ở 0 = dead neuron; phân phối rộng = neuron học đặc trưng đa dạng", fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.hist(act_h1[i], bins=50, color='steelblue', edgecolor='none', alpha=0.85)
    ax.axvline(0, color='crimson', linestyle='--', linewidth=1.2, label='zero')
    ax.set_title(f"Label: {class_names[sample_labels[i].item()]}", fontsize=10)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('./output/mlp/hidden1_activations.png', dpi=150)
print("💾 Đã lưu: mlp_hidden1_activations.png")

# --- 3. WEIGHTS của hidden layer 2 ---
# Cải tiến 6.2: net[5] = Linear(512→256): shape (256, 512) — hàng = 256 neuron H2, cột = 512 neuron H1
# net[3] = Linear(512→512): shape (512, 512)
# Không thể reshape thành ảnh 3×32×32 vì input là 512 (không phải pixel)
# → Dùng imshow trực tiếp để thấy ma trận kết nối giữa 512 neuron lớp 1 và 256 neuron lớp 2
# weights_h2 = model.net[5].weight.detach().cpu().numpy()  # (512, 512)
weights_h2 = model.net[5].weight.detach().cpu().numpy()  # (256, 512)

# Cải tiến 6.2: figsize đổi từ (10,10) → (10,5) cho đúng tỉ lệ ma trận 256×512
# fig, ax = plt.subplots(figsize=(10, 10))
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Hidden Layer 2 Weights — ma trận kết nối (256×512)\n"
             "(đỏ = kết nối dương, xanh = kết nối âm)", fontsize=13)
abs_max = abs(weights_h2).max()
ax.imshow(weights_h2, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, aspect='auto')
ax.set_xlabel("Neuron layer 1 (0–511)")
ax.set_ylabel("Neuron layer 2 (0–255)")
plt.tight_layout()
plt.savefig('./output/mlp/hidden2_weights.png', dpi=150)
print("💾 Đã lưu: mlp_hidden2_weights.png")

# --- 4. ACTIVATIONS của hidden layer 2 với 8 ảnh mẫu ---
# Chạy forward qua net[0:7] = Flatten → Linear → LeakyReLU → BN → Dropout → Linear(512→256) → LeakyReLU
# → thấy đặc trưng bậc cao hơn sau khi qua 2 lớp ẩn
# Cải tiến 5.5: histogram phân phối activation thay vì bar chart theo neuron index
#   net[:7] thay vì net[:5] — [:5] chỉ đến Dropout của H1, không phải activation H2
with torch.no_grad():
    act_h2 = model.net[:7](sample_images).cpu().numpy()   # (8, 256) — cải tiến 6.2

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
fig.suptitle("Activation Distribution — Hidden Layer 2 (sau LeakyReLU)\n"
             "so sánh với H1: H2 thường có phân phối sparse hơn = đặc trưng chuyên biệt hơn", fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.hist(act_h2[i], bins=50, color='darkorange', edgecolor='none', alpha=0.85)
    ax.axvline(0, color='crimson', linestyle='--', linewidth=1.2, label='zero')
    ax.set_title(f"Label: {class_names[sample_labels[i].item()]}", fontsize=10)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('./output/mlp/hidden2_activations.png', dpi=150)
print("💾 Đã lưu: mlp_hidden2_activations.png")

# Cải tiến 3.10: Confusion matrix theo class để xem model hay nhầm nhóm nào với nhóm nào
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_targets = []
all_predictions = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        all_targets.extend(target.cpu().tolist())
        all_predictions.extend(predicted.cpu().tolist())

# Classification report: precision, recall, F1 per class — metric chuẩn trong research
# digits=4: hiển thị 4 chữ số thập phân để đối chiếu paper
print("\nClassification Report (best model):")
print(classification_report(all_targets, all_predictions,
                             target_names=class_names, digits=4))

# sklearn confusion matrix thay cho manual for-loop
# normalize='true': row-normalize sẵn (= recall per class trên đường chéo)
num_classes = len(class_names)
cm_array = sk_confusion_matrix(all_targets, all_predictions, normalize='true')
confusion_matrix_normalized = torch.tensor(cm_array, dtype=torch.float32)

# Giữ lại confusion_matrix (raw counts) để dùng cho per_class_acc bên dưới
confusion_matrix = torch.tensor(
    sk_confusion_matrix(all_targets, all_predictions), dtype=torch.int64
)

# Cải tiến 3.10: seaborn.heatmap thay cho imshow + manual text loop — chuẩn publication
# annot=True: tự điền số vào ô; fmt='.2f': định dạng 2 chữ số thập phân
# linewidths=0.4: đường kẻ giữa các ô giúp dễ đọc
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    confusion_matrix_normalized.numpy(),
    annot=True, fmt='.2f', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names,
    vmin=0.0, vmax=1.0,
    linewidths=0.4, linecolor='lightgray',
    annot_kws={'size': 8},
    ax=ax
)
ax.set_title(f'Confusion Matrix (Row-normalized)  —  Overall Acc: {best_acc:.2f}%',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted label', fontsize=11)
ax.set_ylabel('True label', fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig('./output/mlp/confusion_matrix.png', dpi=150)
print('💾 Đã lưu: confusion_matrix.png')

# Cải tiến 3.10: Per-class accuracy bar chart — chuẩn publication (xuất hiện trong hầu hết paper CIFAR)
# Cho thấy class nào model giỏi/kém, bổ sung thông tin mà confusion matrix tổng hợp không thể hiện rõ
per_class_acc = confusion_matrix.diag().float() / confusion_matrix.sum(dim=1).float() * 100
mean_acc = per_class_acc.mean().item()

fig, ax = plt.subplots(figsize=(10, 5))
# Màu xanh nếu >= mean (tốt), đỏ nếu < mean (yếu) — giúp phát hiện class khó ngay lập tức
colors = ['#2196F3' if v >= mean_acc else '#EF5350' for v in per_class_acc.numpy()]
bars = ax.bar(class_names, per_class_acc.numpy(), color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(mean_acc, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_acc:.1f}%')
# Ghi giá trị lên từng cột
for bar, val in zip(bars, per_class_acc.numpy()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, 115)
ax.set_title('Per-class Accuracy (xanh = trên mean, đỏ = dưới mean)', fontsize=13, fontweight='bold')
ax.set_xlabel('Class', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('./output/mlp/per_class_accuracy.png', dpi=150)
print('💾 Đã lưu: per_class_accuracy.png')
