# Autoencoder (AE)

[← README chính](../README.md)

---

## 1. Lý thuyết

Autoencoder là mạng nơ-ron học cách **nén** dữ liệu vào một không gian ẩn (latent space) chiều thấp hơn, rồi **tái tạo** lại dữ liệu gốc từ biểu diễn nén đó.

Mục tiêu: học được biểu diễn cô đọng nhất của dữ liệu mà vẫn giữ lại thông tin cần thiết để tái tạo.

**Hai thành phần chính:**

- **Encoder** $f_\phi$: nén input $x$ → biểu diễn ẩn $z$ (bottleneck)
- **Decoder** $g_\theta$: giải mã $z$ → output tái tạo $\hat{x}$

**Ứng dụng:** giảm chiều dữ liệu, loại nhiễu (denoising), phát hiện bất thường (anomaly detection), khởi tạo trọng số.

---

## 2. Toán học

### Hàm mất mát — Reconstruction Loss

Autoencoder tối thiểu hóa sai số giữa input gốc $x$ và output tái tạo $\hat{x}$:

$$\hat{x} = g_\theta(f_\phi(x))$$

**MSE Loss** — giả sử nhiễu Gaussian, pixel là giá trị liên tục:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \|x_i - \hat{x}_i\|^2$$

Trong đó $N$ là số pixel (hoặc số chiều của $x$). MSE phạt đều tất cả sai số — pixel sáng hay tối đều được xử lý như nhau.

**BCE Loss** — giả sử pixel độc lập theo phân phối Bernoulli, phù hợp khi pixel $\in [0,1]$:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[x_i \log(\hat{x}_i) + (1 - x_i) \log(1 - \hat{x}_i)\right]$$

BCE phạt mạnh hơn khi model dự đoán sai chiều (ví dụ: pixel thực sự sáng nhưng predict gần 0). Repo dùng MSE cho AE chuẩn, BCE được dùng ở VAE (do decoder kết thúc bằng Sigmoid).

### Bottleneck và thông tin

$z$ có chiều thấp hơn $x$ → encoder **buộc phải** chọn lọc thông tin quan trọng nhất để tái tạo. Đây chính là cơ chế học biểu diễn:

- $\dim(z) \ll \dim(x)$: bottleneck chặt, encoder phải học đặc trưng trừu tượng cốt lõi
- $\dim(z) \geq \dim(x)$: mạng có thể học identity (copy thẳng input sang output) — không học được gì

Với MNIST: $\dim(x) = 784$, $\dim(z) = 32$ — tỉ lệ nén ~24×. Encoder phải chắt lọc thông tin hình dạng chữ số vào 32 số thực.

### Latent Space và sampling

AE chuẩn không áp đặt bất kỳ phân phối nào lên $z$ — chỉ yêu cầu $z$ đủ thông tin để decode lại $x$. Hệ quả:

- Các cluster chữ số **tách biệt** rõ trong latent space (t-SNE thấy rõ) — tốt cho phân loại
- Vùng **trống** giữa các cluster: không có điểm train nào ở đó → decoder không biết phải decode ra gì → ảnh vô nghĩa
- Interpolation thẳng giữa $z_a$ và $z_b$ có thể đi qua vùng trống → ảnh trung gian mờ hoặc nhiễu

**Vì sao không sample tự do được?** AE không có prior — không biết vùng nào của latent space là "hợp lệ". Các cách workaround:

- **Uniform sampling** $z \sim \text{Uniform}(-3, 3)$: phần lớn vùng này nằm ngoài phân phối thực của $z$ → ảnh nhiễu
- **Empirical sampling** $z \sim \mathcal{N}(\hat{\mu}, \hat{\sigma})$: ước lượng mean/std từ latent codes thực, sample từ đó — tốt hơn uniform nhưng vẫn không đảm bảo
- **PCA grid**: traverse dọc theo 2 principal component có phương sai lớn nhất — đi theo hướng dữ liệu thực sự biến thiên, hiệu quả hơn uniform

Tất cả đều kém hơn VAE vì VAE áp đặt prior $\mathcal{N}(0, I)$ qua KL, đảm bảo toàn bộ vùng prior đều có dữ liệu học.

---

## 3. Kiến trúc trong repo

### MNIST (`ae_mnist.py`)

```text
Encoder: 784 → Linear(784→256) → ReLU → Linear(256→64) → ReLU → Linear(64→32)
                                                                        ↓
                                                                    z ∈ ℝ³²
                                                                        ↓
Decoder: Linear(32→64) → ReLU → Linear(64→256) → ReLU → Linear(256→784) → Sigmoid

Tổng tham số: ~276,000
latent_dim = 32
Optimizer: Adam (lr=1e-3)
Loss: MSELoss
Epochs: 50
```

Sigmoid ở cuối decoder đảm bảo output $\in [0,1]$, khớp với pixel MNIST đã chuẩn hóa.

---

[← README chính](../README.md)
