import os
import random
import inspect
import time
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    seed: int = 42                                      # Seed ngẫu nhiên để tái lập kết quả (torch/numpy/random)
    batch_size: int = 100                               # Số ảnh mỗi mini-batch; lớn hơn nhanh hơn nhưng tốn bộ nhớ hơn
    epochs: int = 100                                   # Số lần duyệt hết tập train
    architecture: str = "alexnet"                       # Kiến trúc mạng: "lenet5" | "alexnet"
    eval_only: bool = False                             # True: chỉ evaluation từ checkpoint đã lưu, không train
    lr: float = 0.1                                     # Learning rate cho SGD (bước nhảy cập nhật trọng số)
    momentum: float = 0.9                               # Momentum cho SGD, giúp hội tụ ổn định hơn
    weight_decay: float = 5e-4                          # L2 regularization, hạn chế overfitting
    label_smoothing: float = 0.05                       # Label smoothing cho CrossEntropyLoss (giảm over-confident)
    scheduler_type: str = "onecycle"                    # "onecycle" | "plateau" | "none"
    plateau_factor: float = 0.5                         # Hệ số giảm LR khi dùng ReduceLROnPlateau
    plateau_patience: int = 5                           # Số epoch chờ trước khi giảm LR (plateau)
    plateau_min_lr: float = 1e-5                        # Ngưỡng LR thấp nhất cho plateau scheduler
    early_stopping: bool = True                         # Dừng sớm khi metric validation không cải thiện
    early_stopping_patience: int = 10                   # Số epoch chờ trước khi early stop
    early_stopping_min_delta: float = 1e-4              # Mức cải thiện tối thiểu để reset patience
    grad_clip_max_norm: float = 2.0                     # Gradient clipping (L2 norm) để tránh gradient bùng nổ
    max_norm_linear: float = 4.0                        # Max-norm constraint cho lớp Linear (classifier)
    run_softmax_sanity_check: bool = True               # Kiểm tra phân phối softmax ban đầu (~0.1 mỗi lớp)
    run_overfit_batch_check: bool = True                # Kiểm tra khả năng overfit trên 1 mini-batch
    overfit_batch_steps: int = 100                      # Số bước SGD cho bài test overfit 1 batch
    plot_optim_debug_curves: bool = True                # Vẽ thêm đường cong GradNorm và LR
    data_dir: str = "./data"                            # Thư mục chứa dữ liệu CIFAR-10
    out_dir: str = "./output/cnn"                       # Thư mục gốc lưu checkpoint + hình đánh giá (sẽ tách theo architecture)
    n_bins_calibration: int = 10                        # Số bins cho ECE / reliability diagram
    tsne_max_samples: int = 2000                        # Giới hạn số mẫu cho t-SNE để tránh quá chậm
    gradcam_num_images: int = 10                        # Số ảnh minh họa cho file gradcam_examples.png
    misclassified_max: int = 16                         # Số lỗi sai tự tin cao nhất trong gallery
    feature_map_num_images: int = 10                    # Số ảnh đầu vào được vẽ feature map
    feature_map_num_maps: int = 24                      # Số kênh feature map hiển thị mỗi ảnh
    feature_map_select_mode: str = "topk"               # "topk" (kênh kích hoạt mạnh) | "first" (Map 0..k-1)
    run_saliency_viz: bool = True                       # Vẽ saliency map (data gradient) cho lớp dự đoán
    run_occlusion_viz: bool = True                      # Vẽ occlusion sensitivity heatmap
    run_max_activation_viz: bool = True                 # Truy hồi ảnh kích hoạt mạnh nhất cho một số kênh conv
    saliency_num_images: int = 10                       # Số ảnh minh họa cho saliency_maps.png
    occlusion_num_images: int = 6                       # Số ảnh minh họa cho occlusion_sensitivity.png
    occlusion_patch_size: int = 8                       # Kích thước ô che (px) trong occlusion sensitivity
    occlusion_stride: int = 4                           # Bước trượt ô che (px)
    max_activation_num_channels: int = 8                # Số kênh conv được hiển thị trong max_activating_images.png
    max_activation_topk: int = 4                        # Số ảnh top-k mỗi kênh trong max_activating_images.png
    max_activation_max_samples: int = 2000              # Giới hạn số mẫu quét để tìm ảnh kích hoạt mạnh
    run_feature_inversion_viz: bool = True              # Tái dựng ảnh từ CNN code (penultimate features)
    feature_inversion_num_images: int = 10              # Số ảnh minh họa cho feature_inversion.png
    feature_inversion_steps: int = 1500                 # Số bước tối ưu cho mỗi lần inversion
    feature_inversion_lr: float = 0.01                  # Learning rate khi tối ưu ảnh tái dựng
    feature_inversion_tv_weight: float = 2e-3           # Trọng số total-variation để ảnh mượt hơn
    feature_inversion_l2_weight: float = 1e-4           # Trọng số L2 lên input tái dựng
    feature_inversion_cosine_weight: float = 0.15       # Trọng số cosine-feature loss (ổn định cấu trúc)
    feature_inversion_init_noise_std: float = 0.05      # Độ mạnh nhiễu khởi tạo quanh ảnh gốc (không gian normalized)
    feature_inversion_use_original_init: bool = True    # True: khởi tạo từ ảnh gốc + nhiễu; False: noise thuần


# ─────────────────────────────────────────────────────────────────────────────
# Tái lập kết quả & Thiết bị
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Bật deterministic cho cuDNN (chậm nhẹ, đổi lại tái lập kết quả tốt hơn)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _worker_init(worker_id: int) -> None:
    """Mỗi worker của DataLoader có seed riêng để augmentation độc lập."""
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)


# ─────────────────────────────────────────────────────────────────────────────
# Mô hình
# ─────────────────────────────────────────────────────────────────────────────

class LeNet5(nn.Module):
    """
    LeCun et al. (1998) được điều chỉnh cho input RGB 32×32 (CIFAR-10).
    Với kiến trúc này, Grad-CAM nên lấy ở features[3] (Conv2, map 10x10).
    Nếu lấy ở features[6] thì map chỉ còn 1x1 và heatmap sẽ gần như đồng nhất.

    Layer map (input 3x32x32):
      features[0]  Conv2d(3->6, k=5):      3x32x32  -> 6x28x28
      features[1]  ReLU:                   6x28x28  -> 6x28x28
      features[2]  MaxPool2d(2):           6x28x28  -> 6x14x14
      features[3]  Conv2d(6->16, k=5):     6x14x14  -> 16x10x10   (Grad-CAM target)
      features[4]  ReLU:                   16x10x10 -> 16x10x10
      features[5]  MaxPool2d(2):           16x10x10 -> 16x5x5
      features[6]  Conv2d(16->120, k=5):   16x5x5   -> 120x1x1
      features[7]. ReLU:                   120x1x1  -> 120x1x1
      classifier[0] Flatten:               120x1x1  -> 120
      classifier[1] Linear(120->84):       120      -> 84
      classifier[2] ReLU:                  84       -> 84         (t-SNE hook)
      classifier[3] Linear(84->10):        84       -> 10 logits
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),     # [0] Conv1: học edge/màu cơ bản, 3x32x32 -> 6x28x28
            nn.ReLU(),                          # [1] Phi tuyến sau Conv1
            nn.MaxPool2d(2),                    # [2] Downsample 2x: 6x28x28 -> 6x14x14
            nn.Conv2d(6, 16, kernel_size=5),    # [3] Conv2: đặc trưng bậc cao hơn, 6x14x14 -> 16x10x10
            nn.ReLU(),                          # [4] Phi tuyến sau Conv2
            nn.MaxPool2d(2),                    # [5] Downsample 2x: 16x10x10 -> 16x5x5
            nn.Conv2d(16, 120, kernel_size=5),  # [6] Conv3: nén không gian, 16x5x5 -> 120x1x1
            nn.ReLU(),                          # [7] Activation cuối khối feature
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # [0] 120x1x1 -> 120
            nn.Linear(120, 84),                 # [1] Projection 120 -> 84 (embedding nhỏ)
            nn.ReLU(),                          # [2] Activation áp chót (hook cho t-SNE)
            nn.Linear(84, 10),                  # [3] 10 logits cho CIFAR-10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AlexNetCIFAR(nn.Module):
    """
    AlexNet gọn cho CIFAR-10 (input 32x32): giữ tinh thần AlexNet,
    nhưng điều chỉnh stride/pooling để không làm mất quá nhiều thông tin không gian.

    Layer map (input 3x32x32):
      features[0]   Conv2d(3->64, k=3,p=1):      3x32x32   -> 64x32x32
      features[1]   ReLU:                        64x32x32  -> 64x32x32
      features[2]   MaxPool2d(2):                64x32x32  -> 64x16x16
      features[3]   Conv2d(64->192, k=3,p=1):    64x16x16  -> 192x16x16
      features[4]   ReLU:                        192x16x16 -> 192x16x16
      features[5]   MaxPool2d(2):                192x16x16 -> 192x8x8
      features[6]   Conv2d(192->384, k=3,p=1):   192x8x8   -> 384x8x8
      features[7]   ReLU:                        384x8x8   -> 384x8x8
      features[8]   Conv2d(384->256, k=3,p=1):   384x8x8   -> 256x8x8
      features[9]   ReLU:                        256x8x8   -> 256x8x8
      features[10]  Conv2d(256->256, k=3,p=1):   256x8x8   -> 256x8x8   (Grad-CAM target)
      features[11]  ReLU:                        256x8x8   -> 256x8x8
      features[12]  MaxPool2d(2):                256x8x8   -> 256x4x4
      classifier[0] Flatten:                     256x4x4   -> 4096
      classifier[1] Dropout(0.5)
      classifier[2] Linear(4096->1024)
      classifier[3] ReLU
      classifier[4] Dropout(0.5)
      classifier[5] Linear(1024->512)
      classifier[6] ReLU                                  (t-SNE hook)
      classifier[7] Linear(512->10):            512       -> 10 logits
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Luu y: dung ReLU khong inplace de tuong thich voi full_backward_hook trong Grad-CAM.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),    # [0] Stem conv: 3x32x32 -> 64x32x32
            nn.ReLU(),                                               # [1] Phi tuyến sau stem
            nn.MaxPool2d(kernel_size=2, stride=2),                   # [2] Pool1: 64x32x32 -> 64x16x16

            nn.Conv2d(64, 192, kernel_size=3, padding=1),            # [3] Block2 conv: 64x16x16 -> 192x16x16
            nn.ReLU(),                                               # [4] Phi tuyến block2
            nn.MaxPool2d(kernel_size=2, stride=2),                   # [5] Pool2: 192x16x16 -> 192x8x8

            nn.Conv2d(192, 384, kernel_size=3, padding=1),           # [6] Block3 conv: 192x8x8 -> 384x8x8
            nn.ReLU(),                                               # [7] Phi tuyến block3
            nn.Conv2d(384, 256, kernel_size=3, padding=1),           # [8] Block4 conv: 384x8x8 -> 256x8x8
            nn.ReLU(),                                               # [9] Phi tuyến block4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),           # [10] Block5 conv: 256x8x8 -> 256x8x8
            nn.ReLU(),                                               # [11] Phi tuyến block5
            nn.MaxPool2d(kernel_size=2, stride=2),                   # [12] Pool3: 256x8x8 -> 256x4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                            # [0] 256x4x4 -> 4096
            nn.Dropout(p=0.5),                                       # [1] Regularize FC đầu
            nn.Linear(256 * 4 * 4, 1024),                            # [2] FC1: 4096 -> 1024
            nn.ReLU(),                                               # [3] Phi tuyến FC1
            nn.Dropout(p=0.5),                                       # [4] Regularize FC giữa
            nn.Linear(1024, 512),                                    # [5] FC2: 1024 -> 512
            nn.ReLU(),                                               # [6] Penultimate activation (t-SNE)
            nn.Linear(512, num_classes),                             # [7] Output logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass(frozen=True)
class ArchitectureSpec:
    key: str
    name: str
    description: str
    builder: Callable[[], nn.Module]
    penultimate_layer_path: str
    gradcam_layer_path: str
    inversion_layer_path: str
    conv1_layer_path: str
    feature_map_targets: tuple[tuple[str, str, str], ...]


# Hướng dẫn mở rộng an toàn (không vỡ evaluation/visualization):
# 1) Viết class model mới (vd: ResNetTinyCIFAR) và đánh số layer [0], [1]...
# 2) Thêm 1 entry vào get_arch_registry() với key mới.
# 3) Chọn đúng 4 layer path sau:
#    - penultimate_layer_path: activation ngay trước linear cuối (để trích feature t-SNE)
#    - gradcam_layer_path: conv layer còn kích thước không gian >= 4x4 (để heatmap rõ)
#    - inversion_layer_path: conv layer cuối (mặc định cho feature inversion)
#    - conv1_layer_path: conv đầu tiên (để vẽ filter)
#    - feature_map_targets: các activation muốn visualize dạng feature map
# 4) Đổi Config.architecture = "key_moi" và chạy lại.
#
# Template copy nhanh (tham khảo):
# "resnet_tiny": ArchitectureSpec(
#     key="resnet_tiny",
#     name="ResNetTiny-CIFAR",
#     description="Tiny residual network for 32x32 input",
#     builder=ResNetTinyCIFAR,
#     penultimate_layer_path="head.2",
#     gradcam_layer_path="backbone.6",
#     inversion_layer_path="backbone.8",
#     conv1_layer_path="backbone.0",
#     feature_map_targets=(
#         ("Feature Maps after stem", "feature_maps_stem.png", "backbone.1"),
#         ("Feature Maps after block2", "feature_maps_block2.png", "backbone.6"),
#     ),
# )


def get_arch_registry() -> dict[str, ArchitectureSpec]:
    return {
        "lenet5": ArchitectureSpec(
            key="lenet5",
            name="LeNet-5",
            description="LeCun et al. (1998) adapted for 32x32 RGB",
            builder=LeNet5,
            penultimate_layer_path="classifier.2",
            gradcam_layer_path="features.3",
            inversion_layer_path="features.6",
            conv1_layer_path="features.0",
            feature_map_targets=(
                ("Feature Maps after Conv1 + ReLU", "feature_maps_conv1.png", "features.1"),
                ("Feature Maps after Conv2 + ReLU", "feature_maps_conv2.png", "features.4"),
                ("Feature Maps after Conv3 + ReLU", "feature_maps_conv3.png", "features.7"),
            ),
        ),
        "alexnet": ArchitectureSpec(
            key="alexnet",
            name="AlexNet-CIFAR",
            description="Compact AlexNet variant for 32x32 input",
            builder=AlexNetCIFAR,
            penultimate_layer_path="classifier.6",
            gradcam_layer_path="features.10",
            inversion_layer_path="features.10",
            conv1_layer_path="features.0",
            feature_map_targets=(
                ("Feature Maps after Conv1 + ReLU", "feature_maps_conv1.png", "features.1"),
                ("Feature Maps after Conv2 + ReLU", "feature_maps_conv2.png", "features.4"),
                ("Feature Maps after Conv3 + ReLU", "feature_maps_conv3.png", "features.7"),
                ("Feature Maps after Conv4 + ReLU", "feature_maps_conv4.png", "features.9"),
                ("Feature Maps after Conv5 + ReLU", "feature_maps_conv5.png", "features.11"),
            ),
        ),
    }


def resolve_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """Resolve module path, e.g. "features.3" / "classifier.2"."""
    module: nn.Module = model
    for token in path.split("."):
        if token.isdigit():
            module = module[int(token)]
        else:
            module = getattr(module, token)
    return module


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Dữ liệu
# ─────────────────────────────────────────────────────────────────────────────

# Mean & std theo từng kênh của CIFAR-10 (tiền tính trên tập train)
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2023, 0.1994, 0.2010)


def build_loaders(cfg: Config):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_ds = datasets.CIFAR10(cfg.data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=test_tf)

    print(f"📦 [Data] Train samples : {len(train_ds):,}")
    print(f"📦 [Data] Test samples  : {len(test_ds):,}")
    print(f"📦 [Data] Classes       : {test_ds.classes}")

    use_pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=use_pin_memory, worker_init_fn=_worker_init,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=use_pin_memory, worker_init_fn=_worker_init,
    )
    return train_loader, test_loader, test_ds.classes


# ─────────────────────────────────────────────────────────────────────────────
# Vòng lặp huấn luyện & đánh giá
# ─────────────────────────────────────────────────────────────────────────────

def _top5_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    top5 = logits.topk(5, dim=1).indices
    correct = top5.eq(targets.view(-1, 1).expand_as(top5))
    return correct.any(dim=1).float().sum().item()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    total_epochs: int,
    scheduler_batch=None,
    grad_clip_max_norm: float | None = None,
    max_norm_linear: float | None = None,
):
    model.train()
    total_loss, correct_top1, correct_top5, total = 0.0, 0, 0, 0
    epoch_grad_norm = 0.0
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Theo dõi gradient norm để phát hiện tình trạng bùng nổ/tiêu biến gradient.
        grad_sq_sum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq_sum += p.grad.detach().data.norm(2).item() ** 2
        batch_grad_norm = grad_sq_sum ** 0.5
        epoch_grad_norm += batch_grad_norm

        if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

        optimizer.step()

        if scheduler_batch is not None:
            scheduler_batch.step()

        # Max-norm constraint cho lớp Linear để giảm xu hướng trọng số phình quá lớn.
        if max_norm_linear is not None and max_norm_linear > 0:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = torch.renorm(
                        m.weight.data, p=2, dim=0, maxnorm=max_norm_linear
                    )

        pred = logits.argmax(dim=1)
        total_loss   += loss.item() * x.size(0)
        correct_top1 += (pred == y).sum().item()
        correct_top5 += _top5_accuracy(logits, y)
        total        += y.size(0)

        # Log chi tiết theo batch mỗi 100 batch để theo dõi tiến trình.
        if (batch_idx + 1) % 100 == 0:
            running_loss = total_loss / total
            running_acc  = 100.0 * correct_top1 / total
            print(
                f"    🏋️ [Train] Epoch {epoch:02d}/{total_epochs} "
                f"| Batch {batch_idx+1:4d}/{len(loader)} "
                f"| Batch Loss {loss.item():.4f} "
                f"| Running Loss {running_loss:.4f} "
                f"| Running Top-1 {running_acc:.2f}% "
                f"| GradNorm {batch_grad_norm:.3f}"
            )

    elapsed = time.time() - t0
    avg_loss = total_loss / total
    acc_top1 = 100.0 * correct_top1 / total
    acc_top5 = 100.0 * correct_top5 / total
    avg_grad_norm = epoch_grad_norm / max(len(loader), 1)
    print(
        f"✅ [Train] Epoch {epoch:02d}/{total_epochs} complete "
        f"| Loss {avg_loss:.4f} "
        f"| Top-1 {acc_top1:.2f}% "
        f"| Top-5 {acc_top5:.2f}% "
        f"| GradNorm {avg_grad_norm:.3f} "
        f"| Time {elapsed:.1f}s"
    )
    return avg_loss, acc_top1, acc_top5, avg_grad_norm


@torch.no_grad()
def evaluate(model, loader, criterion, device, split: str = "Test"):
    # Ý nghĩa:
    # - Đánh giá tổng quát trên một split (Test/Final-Test): loss, top-1, top-5.
    # - Thu thập y_true, y_pred, probs để dùng cho toàn bộ biểu đồ phía sau.
    # Cách đánh giá:
    # - Ưu tiên top-1 cho chất lượng phân loại trực tiếp.
    # - Dùng top-5 để xem mô hình có "gần đúng" không (hữu ích khi top-1 thấp).
    # - So loss và accuracy giữa các lần chạy để so độ ổn định mô hình.
    model.eval()
    total_loss, correct_top1, correct_top5, total = 0.0, 0, 0, 0
    all_targets, all_preds, all_probs = [], [], []
    t0 = time.time()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        probs  = torch.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1)

        total_loss   += loss.item() * x.size(0)
        correct_top1 += (pred == y).sum().item()
        correct_top5 += _top5_accuracy(logits, y)
        total        += y.size(0)

        all_targets.extend(y.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    elapsed  = time.time() - t0
    avg_loss = total_loss / total
    acc_top1 = 100.0 * correct_top1 / total
    acc_top5 = 100.0 * correct_top5 / total

    print(
        f"✅ [{split}] Loss {avg_loss:.4f} "
        f"| Top-1 {acc_top1:.2f}% "
        f"| Top-5 {acc_top5:.2f}% "
        f"| Samples {total:,} "
        f"| Time {elapsed:.1f}s"
    )
    return (
        avg_loss, acc_top1, acc_top5,
        np.array(all_targets), np.array(all_preds), np.array(all_probs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hàm hỗ trợ trực quan hóa
# ─────────────────────────────────────────────────────────────────────────────

def denormalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(_STD,  device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def _save(fig, out_dir: str, name: str, dpi: int = 150) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 [Plot] Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Đường cong huấn luyện (loss, top-1, top-5)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(out_dir: str, hist: dict, model_name: str) -> None:
    # Ý nghĩa:
    # - Thể hiện động học học tập theo epoch: train/test loss, train/test top-1, train/test top-5.
    # Cách đánh giá:
    # - Underfit: cả train và test đều thấp, loss cao.
    # - Overfit: train tăng mạnh nhưng test chững/giảm, gap train-test lớn.
    # - Hội tụ tốt: test tăng ổn định và loss test giảm/ổn định.
    print("\n🎨 [Viz] Plotting training curves...")
    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, key, title, ylabel in zip(
        axes,
        [("train_loss", "test_loss"), ("train_top1", "test_top1"), ("train_top5", "test_top5")],
        ["Cross-Entropy Loss", "Top-1 Accuracy", "Top-5 Accuracy"],
        ["Loss", "Accuracy (%)", "Accuracy (%)"],
    ):
        ax.plot(epochs, hist[key[0]], marker="o", linewidth=2, label="Train")
        ax.plot(epochs, hist[key[1]], marker="s", linewidth=2, label="Test")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        # Ghi chú giá trị cuối cùng để đọc nhanh kết quả epoch cuối.
        ax.annotate(
            f"{hist[key[1]][-1]:.2f}",
            xy=(epochs[-1], hist[key[1]][-1]),
            xytext=(5, 5), textcoords="offset points", fontsize=9,
        )

    plt.suptitle(f"Learning Curves — {model_name} on CIFAR-10", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, "training_curves.png")


def plot_optim_debug_curves(out_dir: str, hist: dict, model_name: str) -> None:
    # Ý nghĩa:
    # - Theo dõi độ ổn định tối ưu hóa qua gradient norm và learning rate.
    # Cách đánh giá:
    # - GradNorm tăng đột biến liên tục: dễ bất ổn/dao động gradient.
    # - LR của OneCycle tăng rồi giảm mượt; nếu bất thường cần kiểm tra scheduler.
    if len(hist.get("grad_norm", [])) == 0 or len(hist.get("lr", [])) == 0:
        print("ℹ️ [Viz] Không có dữ liệu grad_norm/lr để vẽ debug curves.")
        return

    print("🎨 [Viz] Plotting optimization debug curves...")
    epochs = np.arange(1, len(hist["grad_norm"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist["grad_norm"], marker="o", linewidth=2, color="darkorange")
    axes[0].set_title("Gradient Norm per Epoch", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L2 Norm")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, hist["lr"], marker="o", linewidth=2, color="teal")
    axes[1].set_title("Learning Rate per Epoch", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("LR")
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Optimization Debug Curves — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "optim_debug_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ma trận nhầm lẫn
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion(out_dir: str, y_true, y_pred, class_names) -> None:
    # Ý nghĩa:
    # - Mỗi hàng là lớp thật, mỗi cột là lớp dự đoán.
    # - Ma trận đã chuẩn hóa theo hàng (row-normalized), nên đường chéo chính là recall theo lớp.
    # Cách đánh giá:
    # - Ô đường chéo càng cao càng tốt.
    # - Ô ngoài đường chéo lớn cho biết cặp lớp bị nhầm thường xuyên.
    print("🎨 [Viz] Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.4, linecolor="lightgray", ax=ax,
        vmin=0.0, vmax=1.0,
    )
    ax.set_title("Confusion Matrix (Row-normalised)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save(fig, out_dir, "confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Độ chính xác theo lớp
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_acc(out_dir: str, y_true, y_pred, class_names) -> None:
    # Ý nghĩa:
    # - Độ chính xác từng lớp (class-wise recall) để thấy lớp nào dễ/khó.
    # Cách đánh giá:
    # - Lớp dưới đường mean là điểm nghẽn chính của mô hình.
    # - Nếu chênh lệch giữa lớp cao nhất và thấp nhất quá lớn -> mô hình học không đồng đều.
    print("🎨 [Viz] Plotting per-class accuracy...")
    cm = confusion_matrix(y_true, y_pred)
    per_class = 100.0 * cm.diagonal() / cm.sum(axis=1)
    mean_acc  = per_class.mean()

    # Sắp xếp giảm dần để dễ đọc và dễ phát hiện lớp yếu.
    order = np.argsort(per_class)[::-1]
    sorted_names = [class_names[i] for i in order]
    sorted_vals  = per_class[order]
    colors = ["#1f77b4" if v >= mean_acc else "#d62728" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(sorted_names, sorted_vals, color=colors)
    ax.axhline(mean_acc, linestyle="--", color="black", linewidth=1.5,
               label=f"Mean: {mean_acc:.1f}%")
    for bar, val in zip(bars, sorted_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class Accuracy (sorted)", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save(fig, out_dir, "per_class_accuracy.png")

    # Log chi tiết theo từng lớp để đối chiếu nhanh trên terminal.
    print("📊 [Per-class accuracy]")
    for name, val in zip(sorted_names, sorted_vals):
        marker = "▲" if val >= mean_acc else "▼"
        print(f"    {marker} {name:<12s} {val:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Đường cong ROC + AUC (one-vs-rest)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(out_dir: str, y_true, probs, class_names) -> None:
    # Ý nghĩa:
    # - ROC one-vs-rest cho từng lớp và Macro AUC toàn cục.
    # - Đo khả năng xếp hạng xác suất dương cao hơn âm theo nhiều ngưỡng.
    # Cách đánh giá:
    # - AUC càng gần 1.0 càng tốt, 0.5 ~ đoán ngẫu nhiên.
    # - So sánh AUC theo lớp để biết lớp nào tách tốt/kém.
    print("🎨 [Viz] Plotting ROC curves...")
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, n_classes))

    auc_scores = {}
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=palette[i], linewidth=1.8,
                label=f"{cls} (AUC={roc_auc:.3f})")

    # AUC trung bình kiểu macro (mỗi lớp trọng số như nhau).
    macro_auc = roc_auc_score(y_bin, probs, average="macro")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"ROC Curves — One-vs-Rest (Macro AUC = {macro_auc:.4f})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _save(fig, out_dir, "roc_curves.png")

    print(f"📈 [ROC] Macro AUC = {macro_auc:.4f}")
    for cls, val in sorted(auc_scores.items(), key=lambda t: t[1], reverse=True):
        print(f"    {cls:<12s} AUC = {val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Đường cong Precision-Recall + Average Precision
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(out_dir: str, y_true, probs, class_names) -> None:
    # Ý nghĩa:
    # - Precision-Recall cho từng lớp + Macro AP.
    # - Đặc biệt hữu ích khi muốn tối ưu precision/recall theo lớp cụ thể.
    # Cách đánh giá:
    # - AP càng cao càng tốt; đường cong càng "ôm góc trên-phải" càng tốt.
    # - Dùng để phát hiện lớp có nhiều false positive hoặc false negative.
    print("🎨 [Viz] Plotting Precision-Recall curves...")
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, n_classes))

    ap_scores = {}
    for i, cls in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        ap_scores[cls] = ap
        ax.plot(rec, prec, color=palette[i], linewidth=1.8,
                label=f"{cls} (AP={ap:.3f})")

    # AP trung bình kiểu macro (cân bằng giữa các lớp).
    macro_ap = np.mean(list(ap_scores.values()))
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision-Recall Curves (Macro AP = {macro_ap:.4f})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _save(fig, out_dir, "pr_curves.png")

    print(f"📈 [PR] Macro AP = {macro_ap:.4f}")
    for cls, val in sorted(ap_scores.items(), key=lambda t: t[1], reverse=True):
        print(f"    {cls:<12s} AP = {val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reliability Diagram + ECE (calibration)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(y_true, probs, n_bins: int = 10) -> float:
    """Expected Calibration Error theo Naeini et al. (2015)."""
    # Ý nghĩa:
    # - ECE đo độ "trung thực" của confidence so với độ đúng thực tế.
    # Cách đánh giá:
    # - ECE càng thấp càng tốt; 0 là hiệu chỉnh hoàn hảo.
    confidence = probs.max(axis=1)
    correctness = (probs.argmax(axis=1) == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() > 0:
            acc_bin  = correctness[mask].mean()
            conf_bin = confidence[mask].mean()
            ece += mask.sum() * abs(acc_bin - conf_bin)
    return ece / len(y_true)


def plot_calibration(out_dir: str, y_true, probs, n_bins: int = 10) -> float:
    # Ý nghĩa:
    # - Reliability diagram: so confidence trung bình và accuracy thực tế theo từng bin.
    # - Histogram confidence: so phân phối độ tự tin của mẫu đúng và mẫu sai.
    # Cách đánh giá:
    # - Cột càng gần đường chéo y=x càng tốt (well-calibrated).
    # - Nếu mẫu sai vẫn tập trung confidence cao -> mô hình over-confident.
    print("🎨 [Viz] Plotting reliability diagram (calibration)...")
    confidence  = probs.max(axis=1)
    correctness = (probs.argmax(axis=1) == y_true).astype(float)

    prob_true_cal, prob_pred_cal = calibration_curve(
        correctness, confidence, n_bins=n_bins, strategy="uniform"
    )
    ece = compute_ece(y_true, probs, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bên trái: reliability diagram (đánh giá calibration trực tiếp theo từng bin).
    ax = axes[0]
    ax.bar(
        prob_pred_cal, prob_true_cal,
        width=1.0 / n_bins, align="center",
        color="steelblue", alpha=0.7, label="Model",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.fill_between(
        [0, 1], [0, 1], [0, 1],
        alpha=0.08, color="grey",
    )
    ax.fill_between(
        prob_pred_cal, prob_pred_cal, prob_true_cal,
        alpha=0.25, color="tomato", label="Calibration gap",
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction Correct")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # Bên phải: histogram độ tự tin (đúng vs sai) để xem mô hình có quá tự tin hay không.
    ax2 = axes[1]
    correct   = correctness.astype(bool)
    incorrect = ~correct
    ax2.hist(confidence[correct],  bins=30, alpha=0.65, color="steelblue", label="Correct", density=True)
    ax2.hist(confidence[incorrect], bins=30, alpha=0.65, color="tomato",    label="Incorrect", density=True)
    ax2.axvline(confidence.mean(), color="black", linestyle="--",
                linewidth=1.4, label=f"Mean conf={confidence.mean():.3f}")
    ax2.set_xlabel("Confidence (max softmax probability)")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence Distribution", fontsize=12, fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Model Calibration Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "calibration.png")
    print(f"📏 [Calibration] ECE = {ece:.4f} (lower is better; 0 = perfect)")
    return ece


# ─────────────────────────────────────────────────────────────────────────────
# 7. t-SNE của đặc trưng lớp áp chót
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_penultimate_features(
    model: nn.Module,
    loader,
    device: torch.device,
    max_samples: int,
    penultimate_layer: nn.Module,
):
    """Trích xuất activation lớp áp chót ngay trước lớp Linear cuối."""
    # Ý nghĩa:
    # - Lấy embedding lớp áp chót để xem mức tách cụm đặc trưng của các lớp.
    # Cách đánh giá:
    # - Embedding tách tốt thường đi kèm phân loại tốt hơn.
    model.eval()
    hook_out: dict = {}

    def _hook(_, __, output):
        hook_out["feat"] = output.detach().cpu()

    handle = penultimate_layer.register_forward_hook(_hook)

    feats, labels = [], []
    collected = 0
    for x, y in loader:
        model(x.to(device))
        feats.append(hook_out["feat"])
        labels.append(y)
        collected += y.size(0)
        if collected >= max_samples:
            break

    handle.remove()
    feats  = torch.cat(feats)[:max_samples].numpy()
    labels = torch.cat(labels)[:max_samples].numpy()
    print(f"🧪 [t-SNE] Extracted {feats.shape[0]} feature vectors of dim {feats.shape[1]}")
    return feats, labels


def plot_tsne(out_dir: str, feats, labels, class_names) -> None:
    # Ý nghĩa:
    # - Giảm chiều embedding (cao chiều) xuống 2D để quan sát trực quan cấu trúc cụm.
    # Cách đánh giá:
    # - Cụm cùng lớp gọn và ít chồng lấn -> đặc trưng học tốt.
    # - Chồng lấn mạnh giữa nhiều lớp -> mô hình dễ nhầm lẫn.
    print("🎨 [Viz] Running t-SNE (this may take ~30s)...")
    t0  = time.time()
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": 40,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": 42,
    }
    tsne_params = inspect.signature(TSNE).parameters
    if "max_iter" in tsne_params:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000

    proj = TSNE(**tsne_kwargs).fit_transform(feats)
    print(f"✅ [t-SNE] Done in {time.time()-t0:.1f}s")

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, cls in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            s=8, alpha=0.55, color=palette[i], label=cls,
        )
    ax.legend(markerscale=2.5, fontsize=9, loc="best")
    ax.set_title(
        f"t-SNE — 84-dim Penultimate Features ({len(feats):,} samples)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    _save(fig, out_dir, "tsne.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Trực quan hóa filter Conv1
# ─────────────────────────────────────────────────────────────────────────────

def plot_first_layer_filters(
    model: nn.Module,
    out_dir: str,
    conv1_layer: nn.Module,
    max_filters: int = 6,
) -> None:
    # Ý nghĩa:
    # - Trực quan kernel Conv1 đã học (cạnh, hướng, màu cơ bản).
    # Cách đánh giá:
    # - Filter đa dạng, có cấu trúc rõ thường tốt hơn filter nhiễu/đồng nhất.
    print("🎨 [Viz] Plotting Conv1 filters...")
    if not isinstance(conv1_layer, nn.Conv2d):
        print("⚠️ [Conv1 filters] Target layer is not Conv2d, skipping.")
        return

    weights = conv1_layer.weight.detach().cpu()
    n = min(max_filters, weights.shape[0])
    cols = n; rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.8))
    axes = np.atleast_1d(axes)
    fig.suptitle(
        f"Conv1 Learned Filters ({weights.shape[1]}ch × {weights.shape[2]}×{weights.shape[3]})",
        fontsize=12, fontweight="bold",
    )
    for i, ax in enumerate(axes):
        if i < n:
            w = weights[i]
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            ax.imshow(np.transpose(w.numpy(), (1, 2, 0)))
            ax.set_title(f"Filter {i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    _save(fig, out_dir, "conv1_filters.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Feature Maps (Conv1 và Conv2)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_feature_maps(
    model: nn.Module, loader, device: torch.device, out_dir: str,
    class_names,
    feature_map_targets: list[tuple[str, str, nn.Module]],
    num_images: int = 4,
    num_maps: int = 8,
    select_mode: str = "topk",
) -> None:
    # Ý nghĩa:
    # - Cho thấy mỗi kênh feature phản ứng thế nào với cùng một ảnh đầu vào.
    # Cách đánh giá:
    # - Map có vùng kích hoạt rõ vào đối tượng là tín hiệu tốt.
    # - Nhiều map gần như trống hoàn toàn có thể gợi ý đặc trưng chưa khai thác tốt.
    print("🎨 [Viz] Plotting feature maps...")
    model.eval()
    # Lấy mẫu phân tầng: lấy 1 ảnh mỗi lớp rồi chọn num_images ảnh đầu.
    class_seen = {}
    for x, y in loader:
        for i in range(len(y)):
            c = int(y[i])
            if c not in class_seen:
                class_seen[c] = x[i]
            if len(class_seen) == len(class_names):
                break
        if len(class_seen) == len(class_names):
            break

    chosen_classes = sorted(class_seen.keys())[:num_images]
    x_sel  = torch.stack([class_seen[c] for c in chosen_classes]).to(device)
    y_sel  = torch.tensor(chosen_classes)
    x_vis  = denormalize_batch(x_sel).cpu()
    captured: dict[str, torch.Tensor] = {}
    handles = []

    for _, _, target_module in feature_map_targets:
        key = str(id(target_module))

        def _make_hook(k):
            def _hook(_, __, output):
                captured[k] = output.detach().cpu()
            return _hook

        handles.append(target_module.register_forward_hook(_make_hook(key)))

    try:
        model(x_sel)
    finally:
        for h in handles:
            h.remove()

    def _draw(fmaps, title, fname):
        show_maps = min(num_maps, fmaps.shape[1])

        # Chọn kênh để hiển thị:
        # - topk: lấy các kênh có activation trung bình lớn nhất (dễ quan sát hơn).
        # - first: giữ hành vi cũ, lấy Map 0..k-1.
        mode = select_mode.strip().lower()
        if mode == "first":
            map_indices = np.arange(show_maps)
        else:
            channel_score = np.abs(fmaps.numpy()).mean(axis=(0, 2, 3))
            map_indices = np.argsort(channel_score)[::-1][:show_maps]

        cols = show_maps + 1
        fig, axes = plt.subplots(num_images, cols,
                                 figsize=(2.2 * cols, 2.4 * num_images))
        axes = np.atleast_2d(axes)
        fig.suptitle(title, fontsize=12, fontweight="bold")

        for i in range(num_images):
            axes[i, 0].imshow(np.transpose(x_vis[i].numpy(), (1, 2, 0)))
            axes[i, 0].set_title(f"Input\n{class_names[y_sel[i].item()]}", fontsize=8)
            axes[i, 0].axis("off")
            for j, ch in enumerate(map_indices):
                ax = axes[i, j + 1]
                m = fmaps[i, ch].numpy()
                ax.imshow(m, cmap="viridis")
                if i == 0:
                    ax.set_title(f"Map {int(ch)}", fontsize=8)
                ax.axis("off")

        plt.tight_layout()
        _save(fig, out_dir, fname)

    for title, fname, target_module in feature_map_targets:
        key = str(id(target_module))
        fmap = captured.get(key)
        if fmap is None or fmap.dim() != 4:
            print(f"⚠️ [Feature maps] Missing/invalid map for {fname}, skipping.")
            continue
        _draw(fmap, title, fname)


@torch.no_grad()
def plot_aggregated_conv_views(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    class_names,
    feature_map_targets_with_path: list[tuple[str, str, str, nn.Module]],
    num_images: int = 4,
) -> None:
    # Ý nghĩa:
    # - Tổng hợp activation theo kênh để thấy "bức tranh tổng thể" của mỗi lớp Conv.
    # - Mean/Max/Energy và Eigen-map (PC1) giúp thấy rõ xu hướng đặc trưng bậc cao mà không bị nhiễu bởi kênh riêng lẻ.
    print("🎨 [Viz] Plotting aggregated conv views (mean/max/energy/eigen + filter montage)...")
    model.eval()

    x_batch, y_batch = next(iter(loader))
    n = min(num_images, x_batch.size(0))
    x_sel = x_batch[:n].to(device)
    y_sel = y_batch[:n]
    x_vis = denormalize_batch(x_sel).cpu()

    captured: dict[str, torch.Tensor] = {}
    handles = []

    for _, _, _, target_module in feature_map_targets_with_path:
        key = str(id(target_module))

        def _make_hook(k):
            def _hook(_, __, output):
                captured[k] = output.detach().cpu()
            return _hook

        handles.append(target_module.register_forward_hook(_make_hook(key)))

    try:
        model(x_sel)
    finally:
        for h in handles:
            h.remove()

    def _norm2d(arr: np.ndarray) -> np.ndarray:
        lo = float(arr.min())
        hi = float(arr.max())
        return (arr - lo) / (hi - lo + 1e-8)

    def _pc1_map(sample_fmap: torch.Tensor) -> np.ndarray:
        # sample_fmap: [C, H, W]
        c, h, w = sample_fmap.shape
        mat = sample_fmap.numpy().reshape(c, h * w).T  # [HW, C]
        mat = mat - mat.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(mat, full_matrices=False)
            pc1 = mat @ vt[0]
            return _norm2d(pc1.reshape(h, w))
        except np.linalg.LinAlgError:
            return np.zeros((h, w), dtype=np.float32)

    for title, fname, path, target_module in feature_map_targets_with_path:
        key = str(id(target_module))
        fmap = captured.get(key)
        if fmap is None or fmap.dim() != 4:
            print(f"⚠️ [Agg maps] Missing/invalid map for {fname}, skipping.")
            continue

        mean_map = fmap.mean(dim=1).numpy()
        max_map = fmap.max(dim=1).values.numpy()
        energy_map = torch.sqrt((fmap ** 2).sum(dim=1) + 1e-8).numpy()

        fig, axes = plt.subplots(n, 5, figsize=(14, 2.6 * n))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"{title} — Aggregated Views", fontsize=12, fontweight="bold")

        for i in range(n):
            inp = np.transpose(x_vis[i].numpy(), (1, 2, 0))
            maps = [
                (None, "Input", inp),
                ("viridis", "Mean", _norm2d(mean_map[i])),
                ("viridis", "Max", _norm2d(max_map[i])),
                ("magma", "Energy", _norm2d(energy_map[i])),
                ("plasma", "Eigen-PC1", _pc1_map(fmap[i])),
            ]

            for j, (cmap, name, arr) in enumerate(maps):
                ax = axes[i, j]
                if cmap is None:
                    ax.imshow(arr)
                    ax.set_ylabel(class_names[y_sel[i].item()], fontsize=8)
                else:
                    ax.imshow(arr, cmap=cmap)
                if i == 0:
                    ax.set_title(name, fontsize=9)
                ax.axis("off")

        plt.tight_layout()
        agg_name = fname.replace("feature_maps_", "aggregated_maps_")
        _save(fig, out_dir, agg_name)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Grad-CAM — theo lớp mục tiêu được cấu hình trong ArchitectureSpec
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradcam(
    model: nn.Module, loader, device: torch.device, out_dir: str,
    class_names,
    target_layer: nn.Module,
    target_layer_name: str,
    num_images: int = 8,
) -> None:
    # Ý nghĩa:
    # - Grad-CAM cho biết vùng ảnh nào tác động mạnh lên quyết định dự đoán.
    # Cách đánh giá:
    # - Heatmap bám đúng vật thể chính -> mô hình "nhìn đúng chỗ".
    # - Heatmap tập trung nền/biên ảnh -> mô hình có thể học lệch ngữ cảnh.
    print(f"🎨 [Viz] Computing Grad-CAM (targeting {target_layer_name})...")
    model.eval()
    x, y = next(iter(loader))
    x = x[:num_images].to(device)
    y = y[:num_images]

    activations: torch.Tensor | None = None
    gradients:   torch.Tensor | None = None

    def _fwd_hook(_, __, output):
        nonlocal activations
        activations = output.clone()

    def _bwd_hook(_, _grad_in, grad_output):
        nonlocal gradients
        gradients = grad_output[0].clone()

    # Hook vào layer đã cấu hình để CAM có độ phân giải không gian phù hợp.
    h1 = target_layer.register_forward_hook(_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        logits = model(x)
        pred   = logits.argmax(dim=1)
        score  = logits[torch.arange(x.size(0), device=device), pred].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
    finally:
        h1.remove(); h2.remove()

    if activations is None or gradients is None:
        print("⚠️ [Grad-CAM] Hooks returned None, skipping.")
        return

    # Global-average-pool gradients theo không gian -> trọng số cho từng kênh.
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cams    = torch.relu((weights * activations).sum(dim=1, keepdim=True))

    # Chuẩn hóa CAM theo từng ảnh để trực quan nhất quán.
    cams_flat = cams.flatten(1)
    cams_min  = cams_flat.min(1)[0].view(-1, 1, 1, 1)
    cams_max  = cams_flat.max(1)[0].view(-1, 1, 1, 1)
    cams      = (cams - cams_min) / (cams_max - cams_min + 1e-8)
    cams      = nn.functional.interpolate(cams, size=(32, 32),
                                          mode="bilinear", align_corners=False)

    imgs = denormalize_batch(x).detach().cpu().numpy()
    cams = cams.detach().cpu().numpy()[:, 0]
    pred = pred.detach().cpu().numpy()

    correct_mask = pred == y.numpy()
    print(
        f"📌 [Grad-CAM] Displayed {num_images} samples | "
        f"Correct: {correct_mask.sum()}/{num_images} "
        f"({100*correct_mask.mean():.0f}%)"
    )

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3.0 * num_images))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Grad-CAM — {target_layer_name}",
                 fontsize=13, fontweight="bold")

    for i in range(num_images):
        img  = np.transpose(imgs[i], (1, 2, 0))
        heat = cams[i]
        correct_str = "✓" if correct_mask[i] else "✗"

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input  T:{class_names[y[i].item()]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heat, cmap="jet")
        axes[i, 1].set_title("CAM (heat)", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img)
        axes[i, 2].imshow(heat, cmap="jet", alpha=0.45)
        axes[i, 2].set_title(
            f"Overlay  P:{class_names[pred[i]]} {correct_str}", fontsize=9
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    _save(fig, out_dir, "gradcam_examples.png")


def plot_guided_gradcam(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    class_names,
    target_layer: nn.Module,
    target_layer_name: str,
    num_images: int = 8,
) -> None:
    # Ý nghĩa:
    # - Guided Grad-CAM = Grad-CAM (vị trí quan trọng) × Guided Backprop (chi tiết biên/texture).
    # - Cho overlay sắc nét hơn Grad-CAM thường, dễ thấy mô hình bám vào chi tiết nào.
    print(f"🎨 [Viz] Computing Guided Grad-CAM (targeting {target_layer_name})...")
    model.eval()
    x, y = next(iter(loader))
    x = x[:num_images].to(device)
    y = y[:num_images]

    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def _fwd_hook(_, __, output):
        nonlocal activations
        activations = output

    def _bwd_hook(_, _grad_in, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    h1 = target_layer.register_forward_hook(_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_bwd_hook)

    x_in = x.clone().detach().requires_grad_(True)
    try:
        logits = model(x_in)
        pred = logits.argmax(dim=1)
        score = logits[torch.arange(x_in.size(0), device=device), pred].sum()
        model.zero_grad(set_to_none=True)
        if x_in.grad is not None:
            x_in.grad.zero_()
        score.backward()
    finally:
        h1.remove(); h2.remove()

    if activations is None or gradients is None or x_in.grad is None:
        print("⚠️ [Guided Grad-CAM] Missing grads/acts, skipping.")
        return

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cams = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cams_flat = cams.flatten(1)
    cams_min = cams_flat.min(1)[0].view(-1, 1, 1, 1)
    cams_max = cams_flat.max(1)[0].view(-1, 1, 1, 1)
    cams = (cams - cams_min) / (cams_max - cams_min + 1e-8)
    cams = nn.functional.interpolate(cams, size=(32, 32), mode="bilinear", align_corners=False)

    guided_bp = torch.relu(x_in.grad)
    guided_bp = guided_bp / (guided_bp.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    guided_gray = guided_bp.mean(dim=1, keepdim=True)
    guided_cam = guided_gray * cams
    guided_cam = guided_cam / (guided_cam.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

    imgs = denormalize_batch(x).detach().cpu().numpy()
    cam_np = cams.detach().cpu().numpy()[:, 0]
    guided_np = guided_cam.detach().cpu().numpy()[:, 0]
    pred_np = pred.detach().cpu().numpy()
    correct_mask = pred_np == y.numpy()

    fig, axes = plt.subplots(num_images, 4, figsize=(13, 3.0 * num_images))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Guided Grad-CAM — {target_layer_name}", fontsize=13, fontweight="bold")

    for i in range(num_images):
        img = np.transpose(imgs[i], (1, 2, 0))
        cam = cam_np[i]
        gcam = guided_np[i]
        correct_str = "✓" if correct_mask[i] else "✗"

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input  T:{class_names[y[i].item()]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam, cmap="jet")
        axes[i, 1].set_title("Grad-CAM", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(gcam, cmap="jet")
        axes[i, 2].set_title("Guided CAM", fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(img)
        axes[i, 3].imshow(gcam, cmap="jet", alpha=0.45)
        axes[i, 3].set_title(
            f"Overlay  P:{class_names[pred_np[i]]} {correct_str}", fontsize=9
        )
        axes[i, 3].axis("off")

    plt.tight_layout()
    _save(fig, out_dir, "guided_gradcam_examples.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Saliency Map (data gradient)
# ─────────────────────────────────────────────────────────────────────────────

def plot_input_saliency(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    class_names,
    num_images: int = 8,
) -> None:
    # Ý nghĩa:
    # - Saliency map đo độ nhạy output theo từng pixel đầu vào.
    # - Đây là dạng "data gradient" cơ bản trong nhóm phương pháp gradient-based.
    print("🎨 [Viz] Computing saliency maps (data gradient)...")
    model.eval()
    x, y = next(iter(loader))
    x = x[:num_images].to(device)
    y = y[:num_images]

    x_in = x.clone().detach().requires_grad_(True)
    logits = model(x_in)
    pred = logits.argmax(dim=1)
    score = logits[torch.arange(x_in.size(0), device=device), pred].sum()

    model.zero_grad(set_to_none=True)
    if x_in.grad is not None:
        x_in.grad.zero_()
    score.backward()

    if x_in.grad is None:
        print("⚠️ [Saliency] Không lấy được gradient theo input, bỏ qua.")
        return

    sal = x_in.grad.detach().abs().max(dim=1)[0]  # [N,H,W]
    sal = sal / (sal.amax(dim=(1, 2), keepdim=True) + 1e-8)

    imgs = denormalize_batch(x).detach().cpu().numpy()
    sal_np = sal.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    correct_mask = pred_np == y.numpy()

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 2.9 * num_images))
    axes = np.atleast_2d(axes)
    fig.suptitle("Input Saliency Maps (data gradient)", fontsize=13, fontweight="bold")

    for i in range(num_images):
        img = np.transpose(imgs[i], (1, 2, 0))
        s = sal_np[i]
        mark = "✓" if correct_mask[i] else "✗"

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input  T:{class_names[y[i].item()]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(s, cmap="hot")
        axes[i, 1].set_title("Saliency", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img)
        axes[i, 2].imshow(s, cmap="hot", alpha=0.45)
        axes[i, 2].set_title(f"Overlay  P:{class_names[pred_np[i]]} {mark}", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    _save(fig, out_dir, "saliency_maps.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Occlusion Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_occlusion_sensitivity(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    class_names,
    num_images: int = 4,
    patch_size: int = 8,
    stride: int = 4,
) -> None:
    # Ý nghĩa:
    # - Trượt một ô che trên ảnh và đo mức sụt giảm xác suất lớp mục tiêu.
    # - Đây là phương pháp perturbation-based, độc lập với gradient nội bộ của model.
    print("🎨 [Viz] Computing occlusion sensitivity maps...")
    model.eval()
    x, y = next(iter(loader))
    x = x[:num_images].to(device)
    y = y[:num_images]
    n, _, h, w = x.shape

    base_logits = model(x)
    base_probs = torch.softmax(base_logits, dim=1)
    pred = base_logits.argmax(dim=1)

    patch_size = max(1, min(patch_size, h, w))
    stride = max(1, stride)
    ys = list(range(0, h - patch_size + 1, stride))
    xs = list(range(0, w - patch_size + 1, stride))
    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)

    heat = torch.zeros((n, h, w), device=device)
    cnt = torch.zeros((n, h, w), device=device)

    for i in range(n):
        target_cls = int(pred[i].item())
        base_p = float(base_probs[i, target_cls].item())

        occluded_batch = []
        locs = []
        for yy in ys:
            for xx in xs:
                x_occ = x[i:i + 1].clone()
                x_occ[:, :, yy:yy + patch_size, xx:xx + patch_size] = 0.0
                occluded_batch.append(x_occ)
                locs.append((yy, xx))

        occ_tensor = torch.cat(occluded_batch, dim=0)
        occ_probs = []
        chunk = 128
        for s in range(0, occ_tensor.size(0), chunk):
            logits_occ = model(occ_tensor[s:s + chunk])
            probs_occ = torch.softmax(logits_occ, dim=1)[:, target_cls]
            occ_probs.append(probs_occ)
        occ_probs = torch.cat(occ_probs, dim=0)

        for p_occ, (yy, xx) in zip(occ_probs, locs):
            drop = max(0.0, base_p - float(p_occ.item()))
            heat[i, yy:yy + patch_size, xx:xx + patch_size] += drop
            cnt[i, yy:yy + patch_size, xx:xx + patch_size] += 1.0

    heat = heat / (cnt + 1e-8)
    heat = heat / (heat.amax(dim=(1, 2), keepdim=True) + 1e-8)

    imgs = denormalize_batch(x).detach().cpu().numpy()
    heat_np = heat.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    correct_mask = pred_np == y.numpy()

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 2.9 * num_images))
    axes = np.atleast_2d(axes)
    fig.suptitle("Occlusion Sensitivity", fontsize=13, fontweight="bold")

    for i in range(num_images):
        img = np.transpose(imgs[i], (1, 2, 0))
        hmap = heat_np[i]
        mark = "✓" if correct_mask[i] else "✗"

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input  T:{class_names[y[i].item()]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(hmap, cmap="jet")
        axes[i, 1].set_title("Occlusion heat", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img)
        axes[i, 2].imshow(hmap, cmap="jet", alpha=0.45)
        axes[i, 2].set_title(f"Overlay  P:{class_names[pred_np[i]]} {mark}", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    _save(fig, out_dir, "occlusion_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Retrieving images that maximally activate a neuron/channel
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_max_activating_images(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    target_layer: nn.Module,
    class_names,
    num_channels: int = 8,
    topk: int = 4,
    max_samples: int = 2000,
) -> None:
    # Ý nghĩa:
    # - Tìm ảnh làm một số kênh conv kích hoạt mạnh nhất.
    # - Giúp hiểu mỗi "neuron/channel" đang nhạy với kiểu mẫu thị giác nào.
    print("🎨 [Viz] Retrieving maximally activating images per channel...")
    model.eval()

    if not isinstance(target_layer, nn.Conv2d):
        print("⚠️ [Max-activation] Target layer không phải Conv2d, bỏ qua.")
        return

    c_out = int(target_layer.out_channels)
    num_channels = max(1, min(num_channels, c_out))
    topk = max(1, topk)

    captured: dict[str, torch.Tensor] = {}

    def _hook(_, __, output):
        captured["act"] = output.detach().cpu()

    handle = target_layer.register_forward_hook(_hook)

    # Mỗi kênh lưu top-k tuple: (score, image_tensor_CHW, class_idx).
    top_per_channel: list[list[tuple[float, torch.Tensor, int]]] = [list() for _ in range(c_out)]
    best_score_per_channel = np.full((c_out,), -np.inf, dtype=np.float32)

    seen = 0
    try:
        for x, y in loader:
            x = x.to(device)
            y_np = y.numpy()
            model(x)
            act = captured.get("act")
            if act is None or act.dim() != 4:
                continue

            x_vis = denormalize_batch(x).detach().cpu()
            # Score theo kênh: mean activation trên không gian feature map.
            score = act.mean(dim=(2, 3)).numpy()  # [B, C]

            for b in range(score.shape[0]):
                for c in range(c_out):
                    s = float(score[b, c])
                    if s > best_score_per_channel[c]:
                        best_score_per_channel[c] = s
                    bucket = top_per_channel[c]
                    if len(bucket) < topk:
                        bucket.append((s, x_vis[b].clone(), int(y_np[b])))
                    else:
                        min_idx = min(range(topk), key=lambda i: bucket[i][0])
                        if s > bucket[min_idx][0]:
                            bucket[min_idx] = (s, x_vis[b].clone(), int(y_np[b]))

            seen += x.size(0)
            if seen >= max_samples:
                break
    finally:
        handle.remove()

    if seen == 0:
        print("⚠️ [Max-activation] Không thu được mẫu nào, bỏ qua.")
        return

    # Chọn các kênh có điểm kích hoạt cực đại cao nhất.
    ch_order = np.argsort(best_score_per_channel)[::-1]
    chosen = [int(c) for c in ch_order[:num_channels] if np.isfinite(best_score_per_channel[c])]
    if len(chosen) == 0:
        print("⚠️ [Max-activation] Không có kênh hợp lệ để hiển thị.")
        return

    fig, axes = plt.subplots(len(chosen), topk, figsize=(2.8 * topk, 2.5 * len(chosen)))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        f"Maximally Activating Images (layer channels, scanned {seen} samples)",
        fontsize=12,
        fontweight="bold",
    )

    for r, ch in enumerate(chosen):
        bucket = sorted(top_per_channel[ch], key=lambda t: t[0], reverse=True)
        for c in range(topk):
            ax = axes[r, c]
            if c < len(bucket):
                s, img, cls = bucket[c]
                ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
                ax.set_title(f"ch{ch} | s={s:.2f}\n{class_names[cls]}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    _save(fig, out_dir, "max_activating_images.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Reconstructing images from CNN codes (feature inversion)
# ─────────────────────────────────────────────────────────────────────────────

def _total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def plot_feature_inversion(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    class_names,
    inversion_layer: nn.Module,
    inversion_layer_name: str,
    num_images: int = 6,
    steps: int = 900,
    lr: float = 0.03,
    tv_weight: float = 2e-3,
    l2_weight: float = 1e-4,
    cosine_weight: float = 0.15,
    init_noise_std: float = 0.08,
    use_original_init: bool = True,
) -> None:
    # Ý nghĩa:
    # - Giữ model cố định, tối ưu trực tiếp ảnh đầu vào để feature ở conv layer cuối khớp feature mục tiêu.
    # - Kết quả cho thấy mức thông tin ảnh còn được bảo toàn trong biểu diễn CNN code.
    print("🎨 [Viz] Reconstructing images from CNN codes (feature inversion)...")
    model.eval()
    x, y = next(iter(loader))
    num_images = max(1, min(num_images, x.size(0)))
    x = x[:num_images].to(device)
    y = y[:num_images]

    feat_holder: dict[str, torch.Tensor] = {}

    def _hook(_, __, output):
        feat_holder["feat"] = output

    h = inversion_layer.register_forward_hook(_hook)

    with torch.no_grad():
        _ = model(x)
        if "feat" not in feat_holder:
            h.remove()
            print("⚠️ [Feature inversion] Không lấy được target feature, bỏ qua.")
            return
        target_feat = feat_holder["feat"].detach()

    if use_original_init:
        init = x.detach() + init_noise_std * torch.randn_like(x)
    else:
        init = init_noise_std * torch.randn_like(x)
    x_opt = nn.Parameter(init)

    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    x_min = (0.0 - mean) / std
    x_max = (1.0 - mean) / std

    optimizer = optim.Adam([x_opt], lr=lr)
    steps = max(1, int(steps))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=max(1e-5, lr * 0.05)
    )

    with torch.no_grad():
        x_opt.clamp_(x_min, x_max)

    try:
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            _ = model(x_opt)
            if "feat" not in feat_holder:
                print("⚠️ [Feature inversion] Hook output rỗng trong lúc tối ưu, dừng sớm.")
                break

            rec_feat = feat_holder["feat"]
            loss_feat_mse = nn.functional.mse_loss(rec_feat, target_feat)
            rec_flat = rec_feat.flatten(start_dim=1)
            tgt_flat = target_feat.flatten(start_dim=1)
            loss_feat_cos = 1.0 - nn.functional.cosine_similarity(rec_flat, tgt_flat, dim=1).mean()
            loss_feat = loss_feat_mse + cosine_weight * loss_feat_cos
            loss_tv = _total_variation(x_opt)
            loss_l2 = (x_opt ** 2).mean()
            loss = loss_feat + tv_weight * loss_tv + l2_weight * loss_l2
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                x_opt.clamp_(x_min, x_max)

            if step == 1 or step % 50 == 0 or step == steps:
                print(
                    f"    [Feature inversion] Step {step:03d}/{steps} "
                    f"| feat={loss_feat.item():.4f} "
                    f"(mse={loss_feat_mse.item():.4f}, cos={loss_feat_cos.item():.4f}) "
                    f"| tv={loss_tv.item():.4f} "
                    f"| l2={loss_l2.item():.4f} "
                    f"| lr={scheduler.get_last_lr()[0]:.5f}"
                )
    finally:
        h.remove()

    x_orig = denormalize_batch(x).detach().cpu().numpy()
    x_rec = denormalize_batch(x_opt.detach()).cpu().numpy()

    fig, axes = plt.subplots(num_images, 3, figsize=(9.8, 2.8 * num_images))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        f"Feature Inversion from Last Conv Features ({inversion_layer_name})",
        fontsize=13,
        fontweight="bold",
    )

    for i in range(num_images):
        img_o = np.transpose(x_orig[i], (1, 2, 0))
        img_r = np.transpose(x_rec[i], (1, 2, 0))
        diff = np.abs(img_o - img_r).mean(axis=2)

        axes[i, 0].imshow(img_o)
        axes[i, 0].set_title(f"Original ({class_names[y[i].item()]})", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_r)
        axes[i, 1].set_title("Reconstructed", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff, cmap="magma")
        axes[i, 2].set_title("Abs diff", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    _save(fig, out_dir, "feature_inversion.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Các mẫu dự đoán sai tự tin cao nhất
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_misclassified_examples(
    model: nn.Module, loader, device: torch.device, out_dir: str,
    class_names, max_items: int = 16,
) -> None:
    # Ý nghĩa:
    # - Tổng hợp các mẫu sai có confidence cao nhất (lỗi nguy hiểm nhất).
    # Cách đánh giá:
    # - Nếu nhiều lỗi rất tự tin, cần kiểm tra bias dữ liệu/kiến trúc/loss.
    # - Cặp (True, Pred) lặp lại nhiều lần cho biết hướng cải tiến ưu tiên.
    print("🎨 [Viz] Collecting misclassified examples...")
    model.eval()
    mistakes = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        wrong = pred != y

        if wrong.any():
            idx   = torch.where(wrong)[0]
            x_vis = denormalize_batch(x[idx]).cpu()
            for i in range(len(idx)):
                mistakes.append((
                    float(conf[idx[i]].cpu()),
                    x_vis[i],
                    int(y[idx[i]].cpu()),
                    int(pred[idx[i]].cpu()),
                ))

    if not mistakes:
        print("✅ [Misclassified] No mistakes found in this batch.")
        return

    mistakes.sort(key=lambda t: t[0], reverse=True)
    mistakes = mistakes[:max_items]
    print(
        f"📌 [Misclassified] Showing top {len(mistakes)} most-confident errors "
        f"| Conf min: {mistakes[-1][0]:.3f} | Conf max: {mistakes[0][0]:.3f}"
    )

    cols = 4
    rows = int(np.ceil(len(mistakes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    axes = np.atleast_2d(axes)
    fig.suptitle("Most Confident Misclassifications", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < len(mistakes):
            conf, img, yt, yp = mistakes[i]
            ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
            ax.set_title(
                f"T: {class_names[yt]}\nP: {class_names[yp]}\nConf: {conf:.2f}",
                fontsize=8,
            )
        ax.axis("off")

    plt.tight_layout()
    _save(fig, out_dir, "misclassified_gallery.png")


# ─────────────────────────────────────────────────────────────────────────────
# Hàm chính
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def main():
    cfg = Config()
    set_seed(cfg.seed)

    arch_registry = get_arch_registry()
    arch_key = cfg.architecture.strip().lower()
    if arch_key not in arch_registry:
        valid = ", ".join(sorted(arch_registry.keys()))
        raise ValueError(f"Unknown architecture='{cfg.architecture}'. Valid: {valid}")
    arch = arch_registry[arch_key]
    cfg.out_dir = os.path.join(cfg.out_dir, arch.key)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = get_device()

    print_section("ENVIRONMENT")
    print(f"  Device        : {device}")
    print(f"  Seed          : {cfg.seed}")
    print(f"  Architecture  : {arch.name} ({arch.key})")
    print(f"  Output dir    : {cfg.out_dir}")
    print(f"  Torch version : {torch.__version__}")

    print_section("DATA LOADING")
    train_loader, test_loader, class_names = build_loaders(cfg)
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches : {len(test_loader)}")

    print_section("MODEL")
    model = arch.builder().to(device)
    n_params = count_parameters(model)
    print(f"  Architecture    : {arch.description}")
    print(f"  Parameters      : {n_params:,}")
    print(f"  Grad-CAM layer  : {arch.gradcam_layer_path}")
    print(f"  Inversion layer : {arch.inversion_layer_path}")
    print(f"  t-SNE layer     : {arch.penultimate_layer_path}")

    penultimate_layer = resolve_module_by_path(model, arch.penultimate_layer_path)
    gradcam_layer = resolve_module_by_path(model, arch.gradcam_layer_path)
    inversion_layer = resolve_module_by_path(model, arch.inversion_layer_path)
    conv1_layer = resolve_module_by_path(model, arch.conv1_layer_path)
    feature_map_targets_with_path = [
        (title, fname, path, resolve_module_by_path(model, path))
        for title, fname, path in arch.feature_map_targets
    ]
    feature_map_targets = [
        (title, fname, module)
        for title, fname, _, module in feature_map_targets_with_path
    ]

    # Clamp tự động cho cấu hình feature-map theo architecture hiện tại.
    # - num_images không thể vượt quá số lớp đang có (đang lấy tối đa 1 ảnh/lớp).
    # - num_maps không nên vượt số channel lớn nhất của các target feature-map layer.
    max_images_allowed = max(1, len(class_names))
    max_maps_allowed = 0
    for _, _, path, target_module in feature_map_targets_with_path:
        out_ch = getattr(target_module, "out_channels", None)

        # Nhiều target là ReLU (vd: features.1), khi đó out_channels nằm ở Conv2d liền trước.
        if not isinstance(out_ch, int):
            tokens = path.split(".")
            if tokens and tokens[-1].isdigit():
                idx = int(tokens[-1])
                if idx > 0:
                    parent_path = ".".join(tokens[:-1])
                    parent_module = (
                        resolve_module_by_path(model, parent_path) if parent_path else model
                    )
                    if isinstance(parent_module, nn.Sequential):
                        prev_module = parent_module[idx - 1]
                        out_ch = getattr(prev_module, "out_channels", None)

        if isinstance(out_ch, int):
            max_maps_allowed = max(max_maps_allowed, out_ch)
    if max_maps_allowed <= 0:
        max_maps_allowed = 1

    if cfg.feature_map_num_images < 1:
        print("⚠️ [Config clamp] feature_map_num_images < 1, tự đặt = 1")
        cfg.feature_map_num_images = 1
    elif cfg.feature_map_num_images > max_images_allowed:
        print(
            f"⚠️ [Config clamp] feature_map_num_images={cfg.feature_map_num_images} "
            f"vượt giới hạn của {arch.key} ({max_images_allowed}), tự giảm xuống {max_images_allowed}"
        )
        cfg.feature_map_num_images = max_images_allowed

    if cfg.feature_map_num_maps < 1:
        print("⚠️ [Config clamp] feature_map_num_maps < 1, tự đặt = 1")
        cfg.feature_map_num_maps = 1
    elif cfg.feature_map_num_maps > max_maps_allowed:
        print(
            f"⚠️ [Config clamp] feature_map_num_maps={cfg.feature_map_num_maps} "
            f"vượt giới hạn của {arch.key} ({max_maps_allowed}), tự giảm xuống {max_maps_allowed}"
        )
        cfg.feature_map_num_maps = max_maps_allowed

    print(
        f"  Feature-maps    : images={cfg.feature_map_num_images} (max {max_images_allowed}), "
        f"maps={cfg.feature_map_num_maps} (max {max_maps_allowed})"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best_path  = os.path.join(cfg.out_dir, "cnn_best.pt")
    best_acc   = 0.0
    should_plot_training_curves = False

    history    = {
        "train_loss": [], "test_loss": [],
        "train_top1": [], "test_top1": [],
        "train_top5": [], "test_top5": [],
        "grad_norm": [], "lr": [],
    }

    if cfg.eval_only:
        print_section("EVALUATION-ONLY MODE")
        print("  Skip training  : True")
        print(f"  Checkpoint     : {best_path}")
        if not os.path.exists(best_path):
            raise FileNotFoundError(
                "eval_only=True nhưng chưa tìm thấy checkpoint. "
                f"Hãy train trước hoặc kiểm tra lại đường dẫn: {best_path}"
            )
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        best_acc = -1.0  # Không có best trong phiên chạy này, sẽ thay bằng final_top1 ở phần tổng kết.
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
        )
        scheduler_key = cfg.scheduler_type.strip().lower()
        scheduler_batch = None
        scheduler_epoch = None
        if scheduler_key == "onecycle":
            scheduler_batch = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                epochs=cfg.epochs,
                steps_per_epoch=len(train_loader),
            )
        elif scheduler_key == "plateau":
            scheduler_epoch = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.plateau_factor,
                patience=cfg.plateau_patience,
                min_lr=cfg.plateau_min_lr,
            )
        elif scheduler_key != "none":
            raise ValueError(
                f"scheduler_type='{cfg.scheduler_type}' không hợp lệ. "
                "Chỉ chấp nhận: onecycle | plateau | none"
            )

        print_section("TRAINING")
        print(f"  Epochs        : {cfg.epochs}")
        print(f"  LR            : {cfg.lr}  Momentum: {cfg.momentum}  WD: {cfg.weight_decay}")
        print(f"  Label smooth  : {cfg.label_smoothing}")
        print(f"  Scheduler     : {scheduler_key}")
        if scheduler_key == "plateau":
            print(
                f"  Plateau cfg   : factor={cfg.plateau_factor} "
                f"patience={cfg.plateau_patience} min_lr={cfg.plateau_min_lr}"
            )
        print(
            f"  Early stopping: {cfg.early_stopping} "
            f"(patience={cfg.early_stopping_patience}, min_delta={cfg.early_stopping_min_delta})"
        )
        print(f"  Grad clip     : {cfg.grad_clip_max_norm}")
        print(f"  MaxNorm linear: {cfg.max_norm_linear}")

        best_val_loss = float("inf")
        early_stop_counter = 0

        if cfg.run_softmax_sanity_check:
            print("\n🧪 [Sanity] Softmax distribution check...")
            model.eval()
            with torch.no_grad():
                x_sanity, _ = next(iter(train_loader))
                x_sanity = x_sanity.to(device)
                probs_sanity = torch.softmax(model(x_sanity), dim=1)
                mean_probs = probs_sanity.mean(dim=0).cpu().numpy()
            print(f"    Mean softmax by class: {np.round(mean_probs, 4)}")
            print(f"    Global mean         : {mean_probs.mean():.4f} (kỳ vọng xấp xỉ 0.1)")

        if cfg.run_overfit_batch_check:
            print("\n🧪 [Sanity] Overfit-1-batch check...")
            x_overfit, y_overfit = next(iter(train_loader))
            x_overfit, y_overfit = x_overfit.to(device), y_overfit.to(device)

            overfit_model = arch.builder().to(device)
            overfit_model.load_state_dict(model.state_dict())
            overfit_opt = optim.SGD(
                overfit_model.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )

            overfit_model.train()
            last_loss = None
            for step in range(cfg.overfit_batch_steps):
                overfit_opt.zero_grad()
                logits_overfit = overfit_model(x_overfit)
                loss_overfit = criterion(logits_overfit, y_overfit)
                loss_overfit.backward()
                overfit_opt.step()
                last_loss = float(loss_overfit.item())
                if (step + 1) % 20 == 0:
                    print(f"    Step {step+1:03d}/{cfg.overfit_batch_steps} | Loss {last_loss:.4f}")

            print(f"    Final overfit loss: {last_loss:.4f} (càng thấp càng tốt)")
            del overfit_model

        for epoch in range(1, cfg.epochs + 1):
            print(f"\n  ── Epoch {epoch}/{cfg.epochs} " + "─" * 40)

            tr_loss, tr_top1, tr_top5, tr_grad = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                cfg.epochs,
                scheduler_batch=scheduler_batch,
                grad_clip_max_norm=cfg.grad_clip_max_norm,
                max_norm_linear=cfg.max_norm_linear,
            )
            te_loss, te_top1, te_top5, _, _, _ = evaluate(
                model, test_loader, criterion, device, split="Test"
            )

            if scheduler_epoch is not None:
                scheduler_epoch.step(te_loss)

            for k, v in zip(
                ["train_loss","test_loss","train_top1","test_top1","train_top5","test_top5"],
                [tr_loss, te_loss, tr_top1, te_top1, tr_top5, te_top5],
            ):
                history[k].append(v)
            history["grad_norm"].append(tr_grad)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            # Phân tích khoảng cách train/test để phát hiện overfitting sớm.
            top1_gap = tr_top1 - te_top1
            print(
                f"📌 [Epoch {epoch} summary] "
                f"Train/Test Top-1 gap = {top1_gap:.2f}pp "
                f"({'overfit' if top1_gap > 5 else 'ok'}) "
                f"| GradNorm = {tr_grad:.3f} "
                f"| LR = {optimizer.param_groups[0]['lr']:.6f}"
            )

            if te_top1 > best_acc:
                best_acc = te_top1
                torch.save(model.state_dict(), best_path)
                print(f"🔥 [Checkpoint] New best = {best_acc:.2f}%  -> {best_path}")

            if cfg.early_stopping:
                if te_loss < (best_val_loss - cfg.early_stopping_min_delta):
                    best_val_loss = te_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(
                        f"⏱️ [EarlyStopping] No val-loss improvement "
                        f"{early_stop_counter}/{cfg.early_stopping_patience}"
                    )
                    if early_stop_counter >= cfg.early_stopping_patience:
                        print("🛑 [EarlyStopping] Triggered. Stop training sớm.")
                        break

        should_plot_training_curves = True

    print_section("FINAL EVALUATION (best checkpoint)")
    if not cfg.eval_only:
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    _, final_top1, final_top5, y_true, y_pred, probs = evaluate(
        model, test_loader, criterion, device, split="Final-Test"
    )
    if cfg.eval_only:
        best_acc = final_top1

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Best Top-1          : {best_acc:6.2f}%        ║")
    print(f"  ║  Final Top-1         : {final_top1:6.2f}%        ║")
    print(f"  ║  Final Top-5         : {final_top5:6.2f}%        ║")
    print(f"  ╚══════════════════════════════════════╝")

    print("\n📋 [Classification Report]")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print_section("VISUALISATIONS")

    if should_plot_training_curves:
        plot_training_curves(cfg.out_dir, history, arch.name)
        if cfg.plot_optim_debug_curves:
            plot_optim_debug_curves(cfg.out_dir, history, arch.name)
    else:
        print("ℹ️ [Viz] Bỏ qua training_curves vì đang ở chế độ eval_only.")
    plot_confusion(cfg.out_dir, y_true, y_pred, class_names)
    plot_per_class_acc(cfg.out_dir, y_true, y_pred, class_names)
    plot_roc_curves(cfg.out_dir, y_true, probs, class_names)
    plot_pr_curves(cfg.out_dir, y_true, probs, class_names)
    ece = plot_calibration(cfg.out_dir, y_true, probs, cfg.n_bins_calibration)

    feats, feat_labels = extract_penultimate_features(
        model, test_loader, device, cfg.tsne_max_samples, penultimate_layer
    )
    plot_tsne(cfg.out_dir, feats, feat_labels, class_names)

    plot_first_layer_filters(model, cfg.out_dir, conv1_layer)
    plot_feature_maps(
        model, test_loader, device, cfg.out_dir, class_names,
        feature_map_targets,
        num_images=cfg.feature_map_num_images,
        num_maps=cfg.feature_map_num_maps,
        select_mode=cfg.feature_map_select_mode,
    )
    plot_aggregated_conv_views(
        model,
        test_loader,
        device,
        cfg.out_dir,
        class_names,
        feature_map_targets_with_path,
        num_images=cfg.feature_map_num_images,
    )
    plot_gradcam(
        model, test_loader, device, cfg.out_dir, class_names,
        target_layer=gradcam_layer,
        target_layer_name=arch.gradcam_layer_path,
        num_images=cfg.gradcam_num_images,
    )
    plot_guided_gradcam(
        model,
        test_loader,
        device,
        cfg.out_dir,
        class_names,
        target_layer=gradcam_layer,
        target_layer_name=arch.gradcam_layer_path,
        num_images=cfg.gradcam_num_images,
    )
    if cfg.run_saliency_viz:
        plot_input_saliency(
            model,
            test_loader,
            device,
            cfg.out_dir,
            class_names,
            num_images=cfg.saliency_num_images,
        )
    else:
        print("ℹ️ [Viz] Bỏ qua saliency map theo cấu hình.")
    if cfg.run_occlusion_viz:
        plot_occlusion_sensitivity(
            model,
            test_loader,
            device,
            cfg.out_dir,
            class_names,
            num_images=cfg.occlusion_num_images,
            patch_size=cfg.occlusion_patch_size,
            stride=cfg.occlusion_stride,
        )
    else:
        print("ℹ️ [Viz] Bỏ qua occlusion sensitivity theo cấu hình.")
    if cfg.run_max_activation_viz:
        plot_max_activating_images(
            model,
            test_loader,
            device,
            cfg.out_dir,
            target_layer=gradcam_layer,
            class_names=class_names,
            num_channels=cfg.max_activation_num_channels,
            topk=cfg.max_activation_topk,
            max_samples=cfg.max_activation_max_samples,
        )
    else:
        print("ℹ️ [Viz] Bỏ qua max-activating images theo cấu hình.")
    if getattr(cfg, "run_feature_inversion_viz", True):
        plot_feature_inversion(
            model,
            test_loader,
            device,
            cfg.out_dir,
            class_names,
            inversion_layer=inversion_layer,
            inversion_layer_name=arch.inversion_layer_path,
            num_images=getattr(cfg, "feature_inversion_num_images", 6),
            steps=getattr(cfg, "feature_inversion_steps", 1500),
            lr=getattr(cfg, "feature_inversion_lr", 0.03),
            tv_weight=getattr(cfg, "feature_inversion_tv_weight", 2e-3),
            l2_weight=getattr(cfg, "feature_inversion_l2_weight", 1e-4),
            cosine_weight=getattr(cfg, "feature_inversion_cosine_weight", 0.15),
            init_noise_std=getattr(cfg, "feature_inversion_init_noise_std", 0.05),
            use_original_init=getattr(cfg, "feature_inversion_use_original_init", True),
        )
    else:
        print("ℹ️ [Viz] Bỏ qua feature inversion theo cấu hình.")
    plot_misclassified_examples(
        model, test_loader, device, cfg.out_dir, class_names,
        max_items=cfg.misclassified_max,
    )

    print_section("SUMMARY")
    feature_map_output_files = [fname for _, fname, _ in arch.feature_map_targets]
    aggregated_output_files = [
        fname.replace("feature_maps_", "aggregated_maps_")
        for fname in feature_map_output_files
    ]
    outputs = [
        "training_curves.png", "confusion_matrix.png", "per_class_accuracy.png",
        "roc_curves.png", "pr_curves.png", "calibration.png",
        "tsne.png", "conv1_filters.png",
        "optim_debug_curves.png",
        *feature_map_output_files,
        *aggregated_output_files,
        "gradcam_examples.png", "guided_gradcam_examples.png",
        "saliency_maps.png", "occlusion_sensitivity.png", "max_activating_images.png",
        "feature_inversion.png",
        "misclassified_gallery.png",
    ]
    print(f"  Output directory : {cfg.out_dir}")
    print(f"  Files generated  : {len(outputs)}")
    for f in outputs:
        path = os.path.join(cfg.out_dir, f)
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"    {exists}  {f}")
    print(f"\n  ECE              : {ece:.4f}")
    print(f"  Best Top-1       : {best_acc:.2f}%")
    print(f"  Final Top-5      : {final_top5:.2f}%")


if __name__ == "__main__":
    main()