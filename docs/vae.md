# Variational Autoencoder (VAE)

[← README chính](../README.md)

---

## 1. Lý thuyết

VAE (Kingma & Welling, 2013) mở rộng Autoencoder bằng cách áp đặt một **phân phối xác suất** lên latent space. Thay vì encode $x$ thành một điểm $z$ cố định, VAE encode thành một **phân phối** $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$.

**Hai mục tiêu đồng thời:**

1. **Tái tạo tốt** — latent code phải chứa đủ thông tin để decode lại $x$
2. **Latent space có cấu trúc** — phân phối posterior $q_\phi(z|x)$ phải gần với prior $p(z) = \mathcal{N}(0, I)$

Kết quả: latent space **liên tục và mịn** → có thể sample tự do từ $\mathcal{N}(0, I)$ để generate ảnh mới.

---

## 2. Toán học

### Framework xác suất

VAE đặt bài toán dưới dạng probabilistic generative model:

- **Prior:** $p(z) = \mathcal{N}(0, I)$ — phân phối ta giả sử latent code tuân theo
- **Decoder (likelihood):** $p_\theta(x|z)$ — xác suất sinh ra $x$ từ $z$, parameterize bởi mạng decoder
- **Posterior thực:** $p(z|x) = \dfrac{p_\theta(x|z)\, p(z)}{p(x)}$ — phân phối $z$ khi đã biết $x$

**Tại sao posterior thực intractable?** Mẫu số $p(x) = \int p_\theta(x|z)\, p(z)\, dz$ đòi hỏi tích phân trên toàn bộ không gian $z$ — không có công thức đóng với mạng nơ-ron phi tuyến.

**Giải pháp — Variational Inference:** Xấp xỉ posterior thực bằng một phân phối đơn giản hơn:

$$q_\phi(z|x) = \mathcal{N}\!\left(\mu_\phi(x),\, \sigma^2_\phi(x) \cdot I\right)$$

Encoder học hai hàm $\mu_\phi(x)$ và $\sigma^2_\phi(x)$ — tham số của phân phối $q$ thay vì trực tiếp encode ra $z$.

### ELBO — Evidence Lower BOund

Mục tiêu: tối đa hóa $\log p(x)$ (log-likelihood của dữ liệu). Khai triển:

$$\log p(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x|z)\right]}_{\text{reconstruction}} - \underbrace{\text{KL}\!\left(q_\phi(z|x) \,\|\, p(z)\right)}_{\text{regularization}} + \underbrace{\text{KL}\!\left(q_\phi(z|x) \,\|\, p(z|x)\right)}_{\geq\, 0}$$

Vì số hạng cuối $\geq 0$, ta có cận dưới:

$$\log p(x) \geq \text{ELBO} = \mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x|z)\right] - \text{KL}\!\left(q_\phi(z|x) \,\|\, p(z)\right)$$

**Ý nghĩa từng số hạng:**

- **Reconstruction term** $\mathbb{E}[\log p_\theta(x|z)]$: khuyến khích encoder nén đủ thông tin để decoder tái tạo lại $x$ tốt — giống AE thông thường
- **KL term** $\text{KL}(q_\phi \| p)$: phạt khi posterior $q$ lệch xa prior $\mathcal{N}(0,I)$ — ép latent space có cấu trúc liên tục, phủ kín prior để có thể sample tự do

Tối đa hóa ELBO ↔ tối thiểu hóa:

$$\mathcal{L} = \mathcal{L}_\text{recon} + \beta \cdot \mathcal{L}_\text{KL}$$

### Reconstruction Loss

Pixel MNIST $\in [0,1]$ → mô hình hóa theo Bernoulli → dùng BCE:

$$\mathcal{L}_\text{recon} = -\sum_i \left[x_i \log(\hat{x}_i) + (1 - x_i) \log(1 - \hat{x}_i)\right]$$

Dùng `reduction='sum'` rồi chia batch để đồng bộ scale với KL (nếu dùng `mean` cho recon nhưng `sum` cho KL thì KL sẽ bị underweight nghiêm trọng).

### KL Divergence — Closed-form

Vì cả $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ và $p(z) = \mathcal{N}(0, I)$ đều là Gaussian, KL tính được analytic — không cần Monte Carlo:

$$\text{KL}\!\left(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, I)\right) = -\frac{1}{2} \sum_j \left(1 + \log \sigma^2_j - \mu^2_j - \sigma^2_j\right)$$

Trong đó $j$ chạy qua từng chiều của $z$. Tại minimum: $\mu_j = 0$, $\sigma^2_j = 1$ → $q = p$ → $\text{KL} = 0$.

Trong code:

```python
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
```

(`logvar` = $\log \sigma^2$ để tránh constraint $\sigma^2 > 0$ khi tối ưu)

### Reparameterization Trick

**Vấn đề:** $z \sim \mathcal{N}(\mu, \sigma^2)$ là phép sampling ngẫu nhiên — không differentiable → gradient không backprop được về encoder (không tính được $\partial z / \partial \mu$ và $\partial z / \partial \sigma$).

**Giải pháp:** Tách randomness ra biến ngoài $\varepsilon$ không có tham số:

$$z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

$z$ vẫn có phân phối $\mathcal{N}(\mu, \sigma^2)$ nhưng là hàm **deterministic** của $(\mu, \sigma, \varepsilon)$ → gradient chảy qua $\mu$ và $\sigma$ về encoder bình thường:

$$\frac{\partial z}{\partial \mu} = 1 \quad \text{(gradient đi thẳng)}, \qquad \frac{\partial z}{\partial \sigma} = \varepsilon \quad \text{(gradient nhân với noise)}$$

### KL per Dimension

Phân tích KL từng chiều để chẩn đoán latent space:

$$\text{KL}_j = -\frac{1}{2}\left(1 + \log\sigma^2_j - \mu^2_j - \sigma^2_j\right)$$

- **Active unit** ($\text{KL}_j > 0.1$): encoder encode thông tin thực sự vào chiều $j$ — $q(z_j|x)$ khác nhau tùy input $x$
- **Collapsed unit** ($\text{KL}_j \approx 0$): posterior xấp xỉ prior $\mathcal{N}(0,1)$ với mọi $x$ → encoder bỏ qua chiều này, decoder không học được gì từ $z_j$

Với `latent_dim=32` trên MNIST: kỳ vọng ~10–15 chiều active vì MNIST thực chất chỉ có ~10 yếu tố biến thiên chính (class).

### Posterior Collapse

Khi KL loss tiến về 0 sớm trong khi reconstruction loss vẫn cao: encoder đã bỏ qua toàn bộ input $x$, chỉ output prior $\mathcal{N}(0,I)$ bất kể $x$ là gì — decoder buộc phải học tái tạo mà không có thông tin từ encoder.

Nguyên nhân: optimizer thấy giảm KL dễ hơn giảm recon loss → ưu tiên collapse encoder trước.

**Cách xử lý:**

- Giảm $\beta$ (ví dụ: $\beta = 0.5$) — giảm áp lực KL để reconstruction được ưu tiên hơn
- **KL Annealing** — tăng $\beta$ từ 0 lên 1 dần theo epoch, cho encoder thời gian học reconstruction trước khi bị regularize
- **Free bits** — không phạt KL nếu $\text{KL}_j < \lambda$, đảm bảo tối thiểu $\lambda$ bit thông tin mỗi chiều

---

## 3. Kiến trúc trong repo

### MNIST (`vae_mnist.py`)

```text
Encoder backbone: 784 → Linear(784→256) → ReLU → Linear(256→64) → ReLU
                                                                       ↓
                                                        ┌──────────────┴──────────────┐
                                                   fc_mu(64→32)              fc_logvar(64→32)
                                                        │                             │
                                                        μ ∈ ℝ³²              log σ² ∈ ℝ³²
                                                        └──────────────┬──────────────┘
                                                             z = μ + σ⊙ε (reparameterize)
                                                                        ↓
Decoder: Linear(32→64) → ReLU → Linear(64→256) → ReLU → Linear(256→784) → Sigmoid

latent_dim = 32
Optimizer: Adam (lr=1e-3)
Loss: BCE (recon) + KL (closed-form)
Epochs: 50
```

**Lúc eval:** `reparameterize()` trả về $\mu$ trực tiếp (không sample) → reconstruction deterministic.

---

## 4. β-VAE

Thêm hệ số $\beta > 1$ vào KL term:

$$\mathcal{L} = \mathcal{L}_\text{recon} + \beta \cdot \mathcal{L}_\text{KL}$$

$\beta > 1$ tăng áp lực KL → encoder buộc phải học **disentangled representation** — mỗi chiều latent kiểm soát một yếu tố biến thiên độc lập (ví dụ: độ nghiêng, độ dày nét bút).

Repo dùng $\beta = 1.0$ (VAE chuẩn).

---

## 5. So sánh với Autoencoder

| | AE | VAE |
| --- | --- | --- |
| Latent $z$ | Vector deterministic | Sample từ $\mathcal{N}(\mu, \sigma^2)$ |
| Prior | Không có | $\mathcal{N}(0, I)$ — áp đặt qua KL |
| Sampling | Phải estimate phân phối | Trực tiếp: $z \sim \mathcal{N}(0, I)$ |
| Latent space | Cluster tách biệt, có vùng trống | Liên tục, mịn, phủ kín prior |
| Interpolation | Có thể tạo ảnh vô nghĩa | Mượt, ảnh trung gian có nghĩa |
| Loss | MSE | BCE + KL (ELBO) |
| Gradient qua sampling | N/A | Reparameterization trick |

---

[← README chính](../README.md)
