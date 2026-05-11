# Convolutional Neural Network (CNN)

[← README chính](../README.md)

---

## 1. Lý thuyết

CNN khai thác cấu trúc không gian của ảnh bằng cách dùng **filter có thể học được** trượt (convolution) qua từng vùng cục bộ. Thay vì kết nối toàn bộ pixel như MLP, mỗi neuron CNN chỉ nhìn vào một **receptive field** nhỏ — giảm tham số và tăng khả năng phát hiện đặc trưng cục bộ (cạnh, góc, texture).

**Ba tính chất quan trọng:**

- **Local connectivity** — mỗi neuron chỉ kết nối với vùng nhỏ của input
- **Weight sharing** — cùng một filter áp dụng cho toàn bộ ảnh → bất biến với vị trí
- **Hierarchical features** — lớp đầu học cạnh/góc đơn giản, lớp sau học hình dạng phức tạp hơn

---

## 2. Toán học

### Phép tích chập (Convolution)

Output feature map tại vị trí $(i, j)$ với filter $W$ kích thước $k \times k$:

$$(I * W)[i,j] = \sum_m \sum_n I[i+m,\, j+n] \cdot W[m,n] + b$$

Trong đó:

- $I$ — ảnh input (hoặc feature map từ lớp trước), kích thước $H \times W$
- $W[m,n]$ — trọng số của filter tại vị trí $(m, n)$, với $m, n \in [0, k)$
- $(i+m,\, j+n)$ — vị trí thực tế trên $I$ mà filter đang "nhìn vào" khi đặt tại $(i,j)$
- $b$ — bias, cộng vào toàn bộ feature map sau convolution

Mỗi filter học một pattern cụ thể (cạnh ngang, cạnh dọc, góc, ...). Trượt filter này qua toàn bộ ảnh và tính tích chập → thu được một feature map phản ánh nơi nào trong ảnh có pattern đó.

Kích thước output sau convolution:

$$H_\text{out} = \left\lfloor\frac{H_\text{in} - k + 2 \cdot \text{pad}}{\text{stride}}\right\rfloor + 1, \qquad W_\text{out} = \left\lfloor\frac{W_\text{in} - k + 2 \cdot \text{pad}}{\text{stride}}\right\rfloor + 1$$

Trong đó:

- $k$ — kích thước filter (kernel size)
- $\text{pad}$ — số pixel zero-padding thêm vào mỗi cạnh (để giữ kích thước spatial hoặc kiểm soát shrinkage)
- $\text{stride}$ — bước nhảy của filter sau mỗi lần tích chập; stride=1 dày đặc, stride=2 bỏ qua 1 ô

### Số tham số của Conv layer

$$\text{params} = (k \times k \times C_\text{in} + 1) \times C_\text{out}$$

Trong đó:

- $k \times k$ — kích thước spatial của một filter
- $C_\text{in}$ — số channel input (ví dụ: ảnh RGB có $C_\text{in}=3$)
- $+1$ — bias của mỗi filter
- $C_\text{out}$ — số filter, mỗi filter cho ra một channel trong output

Mỗi filter có kích thước $k \times k \times C_\text{in}$ (xuyên qua toàn bộ channel input), và toàn bộ $C_\text{out}$ filter dùng chung bộ trọng số đó trên mọi vị trí spatial (**weight sharing**) — đây là lý do CNN tiết kiệm tham số hơn Linear rất nhiều.

Ví dụ: Conv2d(1→6, 5×5) = $(5 \times 5 \times 1 + 1) \times 6 =$ **156 tham số**

So với Linear tương đương trên ảnh 28×28: $(784 + 1) \times 6 = 4{,}710$ tham số → CNN tiết kiệm ~30×.

### Max Pooling

Lấy giá trị lớn nhất trong vùng $k \times k$:

$$P[i,j] = \max_{m,n \in [0,k)} F[i \cdot s + m,\ j \cdot s + n]$$

Trong đó:

- $F$ — feature map đầu vào (thường sau ReLU)
- $s$ — stride của pooling (thường bằng $k$, không overlap)
- $i \cdot s + m,\ j \cdot s + n$ — vị trí thực tế trong $F$ ứng với ô $(i,j)$ của output

Max pooling không có tham số học được. Tác dụng:

- Giảm kích thước spatial: với $k=2$, $s=2$ → $H$, $W$ giảm một nửa, giảm tính toán ở lớp sau
- **Translation invariance**: nếu feature dịch chuyển nhẹ trong vùng $k\times k$, max vẫn giữ nguyên → mạng bớt nhạy cảm với vị trí chính xác của pattern
- Giữ lại giá trị kích hoạt lớn nhất — tức vị trí filter phản ứng mạnh nhất với pattern

---

## 3. Kiến trúc trong repo

### LeNet-5 — MNIST (`cnn_mnist.py`)

Input ảnh MNIST 28×28 được pad thành 32×32 để khớp thiết kế gốc LeNet-5.

```text
Input  : (B, 1, 32, 32)
C1     : Conv2d(1→6,   5×5) → (B,  6, 28, 28)   [156 params]
S2     : MaxPool(2×2)        → (B,  6, 14, 14)
C3     : Conv2d(6→16,  5×5) → (B, 16, 10, 10)   [2,416 params]
S4     : MaxPool(2×2)        → (B, 16,  5,  5)
C5     : Conv2d(16→120,5×5) → (B,120,  1,  1)   [48,120 params]
Flatten:                       (B, 120)
F6     : Linear(120→84)     → (B,  84)           [10,164 params]
Output : Linear(84→10)      → (B,  10)           [850 params]

Tổng: 61,706 tham số
Optimizer: SGD (lr=0.1)
Loss: CrossEntropyLoss
Epochs: 5
```

**Điều chỉnh so với bản gốc:**

- Sigmoid/Tanh → ReLU (gradient ổn định hơn, không bị vanishing)
- AvgPool → MaxPool (giữ đặc trưng nổi bật hơn)
- RBF output units → Linear(84→10) + CrossEntropyLoss

### LeNet-5 — CIFAR-10 (`cnn_cifar10.py`, `architecture="lenet5"`)

Giữ nguyên kiến trúc LeNet-5 nhưng đổi input channel thành 3 (RGB):

```text
Input  : (B, 3, 32, 32)
C1     : Conv2d(3→6,   5×5) → (B,  6, 28, 28)
S2     : MaxPool(2×2)        → (B,  6, 14, 14)
C3     : Conv2d(6→16,  5×5) → (B, 16, 10, 10)
S4     : MaxPool(2×2)        → (B, 16,  5,  5)
C5     : Conv2d(16→120,5×5) → (B,120,  1,  1)
Flatten:                       (B, 120)
F6     : Linear(120→84)     → (B,  84)
Output : Linear(84→10)      → (B,  10)
```

LeNet-5 quá nhỏ cho CIFAR-10 (ảnh màu phức tạp hơn nhiều so với MNIST). Dùng để so sánh với AlexNet.

### AlexNet — CIFAR-10 (`cnn_cifar10.py`, `architecture="alexnet"`)

AlexNet (Krizhevsky et al., 2012) được thu nhỏ cho ảnh 32×32 — giữ tinh thần gốc (nhiều lớp conv sâu + FC lớn + Dropout) nhưng điều chỉnh stride/padding để không làm mất spatial quá sớm:

```text
Input    : (B,   3, 32, 32)
Conv1    : Conv2d(  3→ 64, k=3, p=1) + ReLU → (B,  64, 32, 32)
Pool1    : MaxPool(2×2)                      → (B,  64, 16, 16)
Conv2    : Conv2d( 64→192, k=3, p=1) + ReLU → (B, 192, 16, 16)
Pool2    : MaxPool(2×2)                      → (B, 192,  8,  8)
Conv3    : Conv2d(192→384, k=3, p=1) + ReLU → (B, 384,  8,  8)
Conv4    : Conv2d(384→256, k=3, p=1) + ReLU → (B, 256,  8,  8)
Conv5    : Conv2d(256→256, k=3, p=1) + ReLU → (B, 256,  8,  8)
Pool3    : MaxPool(2×2)                      → (B, 256,  4,  4)
Flatten  :                                     (B, 4096)
Dropout(0.5) + Linear(4096→1024) + ReLU
Dropout(0.5) + Linear(1024→512)  + ReLU
Linear(512→10)

Tổng tham số: ~6,977,000
Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
Scheduler: OneCycleLR
Loss: CrossEntropyLoss (label_smoothing=0.05)
Epochs: 100
```

**So sánh với bản gốc AlexNet (ImageNet, 224×224):**

- Filter size: 11×11/5×5 → **3×3** (ảnh nhỏ hơn nhiều, filter lớn sẽ mất spatial ngay lập tức)
- Stride Conv1: 4 → **1** (stride 4 trên 32×32 cho output 8×8 sau conv1, quá nhỏ)
- LRN (Local Response Normalization) → bỏ (hiệu quả thực tế kém, BN thay thế tốt hơn)
- Dual-GPU split → bỏ (không cần thiết với scale này)

---

## 4. Kỹ thuật tối ưu trên CIFAR-10

**Dropout** — tắt ngẫu nhiên neuron với xác suất $p$ lúc train:

- Mỗi neuron có thể bị tắt độc lập → mạng không thể dựa vào một tập neuron cố định → giảm co-adaptation
- Lúc eval: tắt dropout, nhân output với $(1-p)$ để giữ đúng kỳ vọng (PyTorch tự xử lý)
- AlexNet dùng Dropout(0.5) ở 2 lớp FC đầu — nơi có nhiều tham số nhất

**Data Augmentation** — tăng đa dạng dữ liệu train mà không cần thêm ảnh mới:

- `RandomCrop(32, padding=4)` — pad 4 pixel mỗi cạnh rồi crop ngẫu nhiên về 32×32; model học bất biến với vị trí vật thể
- `RandomHorizontalFlip` — lật ngang; phù hợp CIFAR-10 vì hầu hết class đối xứng (ô tô, chim, ...)

**Label Smoothing** — thay one-hot label cứng bằng phân phối mềm:

$$y_c^\text{smooth} = \begin{cases} 1 - \varepsilon & c = \text{class đúng} \\ \varepsilon / (C-1) & c \neq \text{class đúng} \end{cases}$$

Với $\varepsilon = 0.05$, $C = 10$. Ngăn model quá tự tin (overconfident) vào class đúng → cải thiện calibration và tổng quát hóa.

**Weight Decay** ($\lambda = 5 \times 10^{-4}$) — thêm penalty $\lambda \|W\|^2_2$ vào loss:

$$L_\text{total} = L_\text{CE} + \lambda \sum_l \|W^{(l)}\|^2_2$$

Phạt trọng số lớn → giữ trọng số nhỏ → giảm overfitting. Tương đương L2 regularization, được tích hợp trực tiếp vào SGD qua `weight_decay`.

**OneCycleLR** — lr tăng từ thấp lên đỉnh rồi giảm theo cosine trong một chu kỳ:

- Giai đoạn warm-up (lr tăng): thoát khỏi vùng loss landscape phẳng ban đầu
- Giai đoạn decay (lr giảm): hội tụ tinh tế về minima sắc nét hơn
- Thực tế hội tụ nhanh hơn nhiều so với constant lr hoặc StepLR

**Gradient Clipping** — giới hạn L2 norm của toàn bộ gradient:

$$\text{nếu } \|\nabla\|_2 > \tau: \quad \nabla \leftarrow \frac{\tau}{\|\nabla\|_2} \cdot \nabla$$

Với $\tau = 2.0$. Ngăn gradient exploding làm trọng số nhảy quá xa, đặc biệt hữu ích khi mạng sâu.

**Max Norm Constraint** — sau mỗi bước update, clip L2 norm của từng neuron trong lớp Linear:

$$\text{nếu } \|w_i\|_2 > c: \quad w_i \leftarrow \frac{c}{\|w_i\|_2} \cdot w_i$$

Với $c = 4.0$. Bổ sung cho weight decay — weight decay phạt nhẹ liên tục, max norm là hard constraint tuyệt đối.

---

## 5. Tại sao CNN tốt hơn MLP cho ảnh?

| | MLP | CNN |
| --- | --- | --- |
| Số tham số (MNIST) | 669,706 | 61,706 |
| Khai thác cấu trúc không gian | Không | Có |
| Bất biến với dịch chuyển | Không | Có (qua pooling) |
| Tổng quát hóa | Kém hơn | Tốt hơn |

MLP coi ảnh như vector 1 chiều, mất hoàn toàn thông tin vị trí tương đối giữa các pixel. CNN giữ cấu trúc 2D và học pattern cục bộ — phù hợp bản chất dữ liệu ảnh.

---

[← README chính](../README.md)
