# Multi-Layer Perceptron (MLP)

[← README chính](../README.md)

---

## 1. Lý thuyết

MLP (Multi-Layer Perceptron) là mạng nơ-ron nhân tạo cơ bản nhất. Mỗi neuron nhận toàn bộ đầu ra của lớp trước (fully connected), áp dụng hàm phi tuyến, và truyền kết quả sang lớp tiếp theo.

**Ý tưởng cốt lõi:** Kết hợp nhiều phép biến đổi tuyến tính xen kẽ với hàm phi tuyến để xấp xỉ bất kỳ hàm số liên tục nào (Universal Approximation Theorem).

---

## 2. Toán học

### Forward pass

Với lớp thứ $l$:

$$
\begin{aligned}
z^{(l)} &= W^{(l)} \cdot a^{(l-1)} + b^{(l)} \\
a^{(l)} &= f\!\left(z^{(l)}\right)
\end{aligned}
$$

Trong đó:

- $W^{(l)}$ — ma trận trọng số, kích thước $n_l$ × $n_{l-1}$
- $b^{(l)} \in \mathbb{R}^{n_l}$ — vector bias
- $f$ — hàm kích hoạt (ReLU, LeakyReLU, ...)

### Hàm mất mát — Cross-Entropy

Dùng cho bài toán phân loại nhiều lớp:

$$L = -\sum_c y_c \cdot \log(\hat{y}_c)$$

Với $\hat{y}_c$ là xác suất softmax trên logit $z_c$:

$$\hat{y}_c = \dfrac{e^{z_c}}{\sum_{k} e^{z_k}}$$

Trong đó:

- $c$ — chỉ số class, chạy từ 0 đến $C-1$ (tổng $C$ class)
- $y_c \in \{0, 1\}$ — one-hot label: bằng 1 đúng ở class đúng, 0 ở mọi class còn lại
- $\hat{y}_c$ — xác suất dự đoán cho class $c$

Vì $y_c = 0$ ở mọi class sai, tổng $\sum_c$ thực ra chỉ còn đúng một số hạng — class đúng. Nên loss đơn giản là $-\log(\hat{y}_\text{true})$: phạt nặng khi model tự tin vào class sai.

`nn.CrossEntropyLoss` trong PyTorch tích hợp sẵn LogSoftmax + NLLLoss nên model chỉ cần trả về logit $z$ thô, không cần tự softmax.

### Backpropagation

Mục tiêu: tính gradient của $L$ theo từng tham số để optimizer cập nhật. Áp dụng chain rule ngược từ output về input:

$$
\begin{aligned}
\frac{\partial L}{\partial W^{(l)}} &= \delta^{(l)} \cdot \left(a^{(l-1)}\right)^\top \\
\frac{\partial L}{\partial b^{(l)}} &= \delta^{(l)} \\
\frac{\partial L}{\partial a^{(l-1)}} &= \left(W^{(l)}\right)^\top \cdot \delta^{(l)}
\end{aligned}
$$

Trong đó $\delta^{(l)} = \dfrac{\partial L}{\partial z^{(l)}}$ là **error signal** tại lớp $l$ — đo mức độ mỗi neuron đóng góp vào loss.

- $\delta^{(l)} \cdot (a^{(l-1)})^\top$ — outer product: mỗi phần tử $W_{ij}$ nhận gradient bằng $\delta^{(l)}_i$ nhân $a^{(l-1)}_j$
- $(W^{(l)})^\top \cdot \delta^{(l)}$ — truyền error signal về lớp trước để tiếp tục backprop

$\delta^{(l)}$ được tính đệ quy từ lớp trên: $\delta^{(l)} = \left((W^{(l+1)})^\top \delta^{(l+1)}\right) \odot f'\!\left(z^{(l)}\right)$, với $f'$ là đạo hàm hàm kích hoạt.

### SGD với Momentum

SGD thuần cập nhật thẳng theo gradient tại bước hiện tại — dễ dao động, chậm qua vùng phẳng. Momentum tích lũy hướng di chuyển qua các bước:

$$
\begin{aligned}
v &\leftarrow \beta \cdot v - \text{lr} \cdot \nabla_W L \\
W &\leftarrow W + v
\end{aligned}
$$

Trong đó:

- $v$ — velocity: trung bình có trọng số của các gradient quá khứ
- $\beta \in [0, 1)$ — hệ số momentum (thường 0.9): quyết định bao nhiêu lịch sử được giữ lại
- $\text{lr}$ — learning rate
- $\nabla_W L$ — gradient của loss theo $W$ tại bước hiện tại

Khi gradient liên tục cùng hướng, $v$ tích lũy → bước đi lớn dần (tăng tốc). Khi gradient đổi chiều, $v$ bị triệt tiêu dần → giảm dao động.

### Hàm kích hoạt

Không có hàm phi tuyến, nhiều lớp Linear chồng lên nhau vẫn chỉ là một phép biến đổi tuyến tính duy nhất — mạng sẽ không học được gì phức tạp hơn.

**ReLU** (Rectified Linear Unit):

$$f(x) = \max(0,\, x), \qquad \frac{\partial f}{\partial x} = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

Gradient bằng 1 ở vùng dương → không bị vanishing khi backprop qua nhiều lớp. Nhưng nếu $z < 0$ liên tục, $\partial f / \partial x = 0$ → gradient bằng 0 → neuron không bao giờ cập nhật (**Dead Neuron**).

**LeakyReLU:**

$$f(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases} \quad (\alpha = 0.01\text{–}0.3)$$

Cho gradient nhỏ $\alpha$ chảy qua vùng âm → neuron không bao giờ chết hoàn toàn.

---

## 3. Kiến trúc trong repo

### MNIST (`mlp_mnist.py`)

```text
Input (784) → Linear(784→512) → ReLU → Linear(512→512) → ReLU → Linear(512→10)
Tổng tham số: 669,706
Optimizer: SGD (lr=0.1)
Loss: CrossEntropyLoss
```

### CIFAR-10 (`mlp_cifar10.py`)

```text
Input (3072) → Linear(3072→512) → LeakyReLU(0.2) → BN → Dropout(0.2)
             → Linear(512→256)  → LeakyReLU(0.3) → BN → Dropout(0.1)
             → Linear(256→10)
Tổng tham số: ~1,707,274
Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)
Scheduler: OneCycleLR
Loss: CrossEntropyLoss (label_smoothing=0.05)
```

**Các kỹ thuật tối ưu trên CIFAR-10:**

**Batch Normalization** — chuẩn hóa activation trước (hoặc sau) mỗi lớp:

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma^2_B + \varepsilon}}, \qquad \text{BN}(z_i) = \gamma \hat{z}_i + \beta$$

Trong đó $\mu_B$, $\sigma^2_B$ là mean/variance của mini-batch, $\gamma$ và $\beta$ là tham số học được (scale và shift). Tác dụng: giữ activation ở vùng gradient lớn của hàm kích hoạt → ổn định và tăng tốc hội tụ, giảm nhạy cảm với learning rate.

**Dropout** — tắt ngẫu nhiên neuron với xác suất $p$ lúc train:

- Mỗi neuron được giữ với xác suất $(1-p)$, tắt thành 0 với xác suất $p$
- Lúc eval: tắt dropout, nhân output với $(1-p)$ để giữ đúng kỳ vọng (PyTorch tự xử lý)
- MLP CIFAR-10 dùng Dropout(0.2) sau lớp 1 và Dropout(0.1) sau lớp 2

**He Initialization** — khởi tạo trọng số phù hợp với ReLU/LeakyReLU:

$$W \sim \mathcal{N}\!\left(0,\, \frac{2}{n_\text{in}}\right)$$

Với $n_\text{in}$ là số neuron đầu vào của lớp. ReLU triệt tiêu ~50% activation → variance tín hiệu giảm một nửa qua mỗi lớp. He init bù lại bằng cách nhân variance lên 2 → tín hiệu không bị tắt dần khi đi qua mạng sâu.

**Data Augmentation** — tăng đa dạng dữ liệu train mà không cần thêm ảnh mới:

- `RandomCrop(32, padding=4)` — pad 4 pixel mỗi cạnh rồi crop ngẫu nhiên về 32×32; model học bất biến với vị trí
- `RandomHorizontalFlip` — lật ngang; phù hợp CIFAR-10 vì hầu hết class đối xứng

**OneCycleLR** — lr tăng từ thấp lên đỉnh rồi giảm theo cosine trong một chu kỳ:

- Giai đoạn warm-up (lr tăng): thoát khỏi vùng loss landscape phẳng ban đầu
- Giai đoạn decay (lr giảm): hội tụ tinh tế về minima sắc nét hơn
- Thực tế hội tụ nhanh hơn nhiều so với constant lr hoặc StepLR

**Gradient Clipping** — giới hạn L2 norm của toàn bộ gradient:

$$\text{nếu } \|\nabla\|_2 > \tau: \quad \nabla \leftarrow \frac{\tau}{\|\nabla\|_2} \cdot \nabla$$

Với $\tau = 2.0$. Ngăn gradient exploding làm trọng số nhảy quá xa.

**Max Norm Constraint** — sau mỗi bước update, clip L2 norm của từng neuron:

$$\text{nếu } \|w_i\|_2 > c: \quad w_i \leftarrow \frac{c}{\|w_i\|_2} \cdot w_i$$

Với $c = 4.0$. Bổ sung cho weight decay — weight decay phạt nhẹ liên tục, max norm là hard constraint tuyệt đối.

---

[← README chính](../README.md)
