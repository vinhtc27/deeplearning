# Long Short-Term Memory (LSTM)

[← README chính](../README.md)

---

## 1. Lý thuyết

LSTM là kiến trúc mạng hồi tiếp (recurrent) được thiết kế để xử lý dữ liệu tuần tự. Khác với MLP hay CNN chỉ nhìn input hiện tại, LSTM duy trì **trạng thái ẩn** (hidden state) từ bước trước — cho phép thông tin từ quá khứ ảnh hưởng đến dự đoán hiện tại.

**Vấn đề của RNN thuần:** Vanishing gradient khi backprop qua nhiều bước thời gian — gradient tắt dần theo cấp số nhân, khiến mạng không thể học quan hệ xa.

**Giải pháp của LSTM:** Tách biệt hai luồng thông tin:

- **Cell state** $c_t$ — bộ nhớ dài hạn, truyền thẳng qua các bước bằng phép nhân và cộng (không qua activation) → gradient chảy trực tiếp, không bị ép qua tanh/sigmoid liên tục
- **Hidden state** $h_t$ — output ngắn hạn, được lọc qua output gate từ $c_t$

Ba **gate** (forget, input, output) học cách kiểm soát thông tin nào được ghi, giữ, và xuất ra — tất cả đều phụ thuộc vào input hiện tại và hidden state trước.

**Ứng dụng:** language modeling, machine translation, time-series forecasting, sequence generation, speech recognition.

---

## 2. Toán học

### RNN thuần

RNN (Recurrent Neural Network) xử lý sequence bằng cách duy trì một hidden state $h_t$ — vector tóm tắt toàn bộ thông tin đã thấy từ bước 1 đến bước $t$. Tại mỗi bước, hidden state được cập nhật từ input hiện tại và hidden state trước:

$$h_t = \tanh\!\left(W_h h_{t-1} + W_x x_t + b\right)$$

Trong đó:

- $W_h \in \mathbb{R}^{H \times H}$ — ma trận trọng số hồi tiếp (recurrent weights), **dùng chung ở mọi bước**
- $W_x \in \mathbb{R}^{H \times d}$ — ma trận chiếu input
- $h_t \in \mathbb{R}^H$ — hidden state tại bước $t$, đóng vai trò bộ nhớ ngắn hạn
- $h_0 = \mathbf{0}$ — khởi tạo bằng vector 0 trước khi đọc sequence

Output tại bước $t$ (nếu cần, ví dụ trong sequence-to-sequence):

$$\hat{y}_t = W_y h_t + b_y$$

**Weight sharing qua thời gian:** $W_h$, $W_x$ được dùng lại ở mọi bước — RNN chỉ có một bộ tham số duy nhất bất kể sequence dài bao nhiêu. Đây cũng là lý do gradient cần được tính qua toàn bộ $T$ bước (BPTT).

**Unrolled computation graph** — minh họa qua 4 bước:

```text
x₁ → [W_x]──►[tanh]──h₁──►[W_x]──►[tanh]──h₂──► ...
                ▲                ▲
               [W_h]            [W_h]
                │                │
               h₀               h₁
```

Mỗi ô `[tanh]` dùng chung $W_h$ và $W_x$ — unrolling chỉ là cách nhìn đồ thị tính toán để áp dụng backprop.

### Vanishing gradient trong RNN

Gradient của loss $L$ tại bước $T$ đối với hidden state ở bước $t$ qua chain rule:

$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

Jacobian của một bước hồi tiếp:

$$\frac{\partial h_{k+1}}{\partial h_k} = W_h^\top \cdot \text{diag}\!\left(\tanh'\!(z_{k+1})\right) = W_h^\top \cdot \text{diag}\!\left(1 - h_{k+1}^2\right)$$

Trong đó $z_{k+1} = W_h h_k + W_x x_{k+1} + b$ là pre-activation tại bước $k+1$.

Tích qua $T - t$ bước:

$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t}^{T-1} W_h^\top \cdot \text{diag}\!\left(1 - h_{k+1}^2\right)$$

**Hai vấn đề:**

**1. Vanishing gradient** — $\tanh'(x) = 1 - \tanh^2(x) \in (0, 1]$, đạt cực đại bằng 1 chỉ tại $x = 0$. Khi hidden state bão hòa ($|h| \approx 1$), $\tanh' \approx 0$. Mỗi bước nhân thêm một hệ số $< 1$ → tích qua $T - t$ bước tắt dần theo cấp số nhân → gradient từ bước xa về gần 0.

Điều kiện để không vanish: phải có $\|W_h^\top \cdot \text{diag}(1-h^2)\|_2 \geq 1$ tại mọi bước — thực tế rất khó đảm bảo.

**2. Exploding gradient** — Nếu $\|W_h\|_2 > 1$ và tanh chưa bão hòa, tích trên có thể tăng theo cấp số nhân → gradient bùng nổ → trọng số nhảy xa, training phân kỳ. Giải pháp thực tế: gradient clipping.

**Hệ quả thực tế:** RNN thuần chỉ học được quan hệ ngắn hạn (~5–10 bước). Quan hệ xa hơn (ví dụ: chủ ngữ và động từ cách nhau 20 từ) thực tế không học được.

### LSTM Cell — các phương trình

Tại mỗi bước thời gian $t$, LSTM nhận $x_t \in \mathbb{R}^d$ và trạng thái trước $(h_{t-1}, c_{t-1})$, xuất ra trạng thái mới $(h_t, c_t)$:

$$\begin{bmatrix} f_t \\ i_t \\ g_t \\ o_t \end{bmatrix} = \begin{bmatrix} \sigma \\ \sigma \\ \tanh \\ \sigma \end{bmatrix} \!\left(W \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b\right)$$

Trong đó $W \in \mathbb{R}^{4H \times (H+d)}$ và $b \in \mathbb{R}^{4H}$ là ma trận trọng số và bias gộp chung cho cả 4 gates.

**Cell state update:**

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

**Hidden state:**

$$h_t = o_t \odot \tanh(c_t)$$

Trong đó:

- $H$ — kích thước hidden state
- $\sigma$ — hàm sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$
- $\odot$ — element-wise multiplication (Hadamard product)
- $[h_{t-1},\, x_t]$ — vector ghép (concatenate) kích thước $H + d$

### Ý nghĩa từng gate

**Forget gate** $f_t \in (0, 1)^H$:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

Kiểm soát bao nhiêu phần của $c_{t-1}$ được giữ lại. $f_t \approx 1$ → giữ nguyên bộ nhớ cũ. $f_t \approx 0$ → xóa. Trong $c_t = f_t \odot c_{t-1} + \ldots$, đây là phép nhân vô hướng không qua activation — gradient chảy thẳng về quá khứ.

**Input gate** $i_t \in (0, 1)^H$:

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

Kiểm soát bao nhiêu thông tin mới từ $g_t$ được ghi vào $c_t$.

**Cell candidate** $g_t \in (-1, 1)^H$:

$$g_t = \tanh(W_g [h_{t-1}, x_t] + b_g)$$

Nội dung cần ghi vào cell state — $i_t \odot g_t$ là phần thực sự được cộng vào $c_t$.

**Output gate** $o_t \in (0, 1)^H$:

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

Kiểm soát bao nhiêu cell state lộ ra ngoài thành hidden state. $h_t = o_t \odot \tanh(c_t)$ — tanh ép $c_t$ về $(-1, 1)$, output gate lọc neuron nào được dùng tại bước này.

### Tại sao LSTM giải quyết vanishing gradient

Gradient của loss $L$ đối với $c_{t-1}$ qua bước update $c_t = f_t \odot c_{t-1} + i_t \odot g_t$:

$$\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t)$$

Gradient chảy về qua phép nhân với $f_t$ — không qua activation nào. Nếu forget gate học giữ $f_t \approx 1$ cho một neuron, gradient của neuron đó truyền nguyên vẹn về bao nhiêu bước cũng được. So với RNN thuần phải nhân với $W_h^\top \cdot \text{diag}(1 - h^2)$ tại mỗi bước — sự khác biệt là quyết định.

Gradient đầy đủ qua $T$ bước với LSTM (đơn giản hóa):

$$\frac{\partial L}{\partial c_t} = \frac{\partial L}{\partial c_T} \prod_{k=t}^{T-1} f_{k+1}$$

Tích này chỉ vanish nếu forget gate học ra giá trị nhỏ — hoàn toàn do dữ liệu quyết định, không phải do kiến trúc ép buộc.

**Per-neuron memory:** $f_t \in (0,1)^H$ là vector — mỗi neuron có forget gate riêng. Neuron $i$ có thể nhớ 200 bước ($f_{t,i} \approx 1$) trong khi neuron $j$ quên ngay ($f_{t,j} \approx 0$). RNN không có cơ chế này — toàn bộ hidden state bị ép qua cùng một $\tanh$.

### Hàm mất mát — MSE cho Patch Prediction

Bài toán dự đoán patch tiếp theo là bài toán **regression trên pixel** (không phải classification). Loss tối thiểu hóa sai số pixel giữa predicted và actual patch tại mỗi bước $t$:

$$\mathcal{L} = \frac{1}{(T-1) \cdot P} \sum_{t=1}^{T-1} \sum_{p=1}^{P} \left(\hat{x}_{t+1,p} - x_{t+1,p}\right)^2$$

Trong đó:

- $T = 16$ — tổng số patch (4×4 grid)
- $P = 49$ — số pixel mỗi patch (7×7)
- $\hat{x}_{t+1}$ — patch được predict sau khi đọc patch $t$ (teacher forcing: input luôn là actual patch $t$)
- $x_{t+1}$ — actual patch $t+1$ (ground truth)

**PSNR** (Peak Signal-to-Noise Ratio) — độ đo dễ đọc hơn MSE:

$$\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{1}{\text{MSE}}\right) \text{ (dB)}$$

Với pixel $\in [0,1]$. MSE = 0.029 → PSNR ≈ 15.4 dB. Ảnh tự nhiên chất lượng cao thường >30 dB; MNIST đơn giản hơn nhưng task prediction khó hơn reconstruction (AE) nên giá trị thấp hơn là bình thường.

### Teacher Forcing vs Autoregressive Inference

**Teacher forcing (train):** tại mỗi bước $t$, feed actual patch $x_t$ làm input dù prediction bước trước đúng hay sai:

$$h_t, c_t = \text{LSTMCell}(x_t,\, h_{t-1},\, c_{t-1})$$

Gradient rõ ràng tại từng bước → hội tụ ổn định hơn.

**Autoregressive inference (test):** sau khi warm up với $k$ patch ground truth, dùng output làm input tiếp theo:

$$\hat{x}_{t+1} = \text{head}(h_t), \quad h_{t+1}, c_{t+1} = \text{LSTMCell}(\hat{x}_{t+1},\, h_t,\, c_t)$$

**Exposure bias:** mô hình train trên input thật, test trên input predict → phân phối input khác nhau giữa hai giai đoạn. Lỗi tích lũy theo bước: prediction sai ở bước $t$ → input sai ở bước $t+1$ → prediction sai hơn. Giải pháp: *scheduled sampling* — trộn dần input thật và predicted trong training.

---

## 3. Kiến trúc trong repo

### MNIST — Next Patch Prediction (`lstm_mnist.py`)

Task: cắt ảnh 28×28 thành 4×4 grid (16 patch 7×7). LSTM đọc patch theo thứ tự raster (trái→phải, trên→dưới). Tại mỗi bước $t$, predict pixel của patch $t+1$.

**Patchify:**

```text
(B, 1, 28, 28)
  → unfold(H, 7, 7) → (B, 1, 4, 28, 7)
  → unfold(W, 7, 7) → (B, 1, 4, 4, 7, 7)
  → view            → (B, 16, 49)
```

**Kiến trúc:**

```text
Input patch x_t ∈ ℝ⁴⁹, hidden/cell state ∈ ℝ²⁵⁶

LSTMCell:
  Linear(49+256 → 4×256)  →  chunk(4)  →  f, i, g, o
  c_t = f ⊙ c_{t-1} + i ⊙ g
  h_t = o ⊙ tanh(c_t)

Head: Linear(256 → 49) → Sigmoid   (pixel values ∈ [0,1])

Tham số:
  LSTMCell.linear : (49 + 256) × (4 × 256) + (4 × 256) = 313,344
  head.linear     :  256 × 49 + 49          =  12,593
  Tổng: 325,937

Optimizer  : Adam (lr=1e-3)
Loss       : MSELoss
Metric     : PSNR (dB)
Epochs     : 20
```

**Linear gộp 4 gates:** thay vì 4 Linear riêng, dùng 1 Linear có output $4H$ rồi `chunk(4)` cắt ra — kết quả toán học giống hệt nhưng 1 matmul lớn nhanh hơn 4 matmul nhỏ trên GPU.

**Inference với `num_given` patch:**

```text
Warm up: feed patch 0, 1, ..., num_given-1 (ground truth)
         → h, c tích lũy context nửa trên ảnh

Autoregressive:
  pred = head(h)          ← predict patch num_given
  loop:
    h, c = LSTMCell(pred, h, c)
    pred = head(h)        ← predict patch tiếp theo
```

**4 Visualizations:**

| File | Nội dung |
| ---- | -------- |
| `error_analysis.png` | MSE heatmap 4×4 (vị trí khó predict) + temporal line (teacher forcing) |
| `step_by_step.png` | Progressive completion autoregressive (xanh = given, đỏ = predicted) |
| `lstm_internals.png` | Gate dynamics (f/i/g/o mean±std per step) + h_t/c_t heatmaps |
| `forget_analysis.png` | Forget gate spatial 4×4 + mean forget per digit class |

---

## 4. Giới hạn và hướng mở rộng

**MSE → blurry prediction:** MSE tối thiểu tại giá trị trung bình của tất cả khả năng. Nếu patch tiếp theo có thể là nét đậm hoặc khoảng trắng với xác suất bằng nhau, model predict màu xám trung gian — không sắc nét. Giải pháp: perceptual loss, GAN discriminator, hoặc diffusion model.

**Exposure bias:** teacher forcing tạo gap giữa train và inference. *Scheduled sampling* trộn dần input thật và predicted trong quá trình train để thu hẹp gap.

**Thứ tự patch tuyến tính:** LSTM đọc patch theo raster order — không có khái niệm vị trí 2D. Patch (1,0) không biết nó nằm ngay dưới patch (0,0). **Vision Transformer (ViT)** giải quyết bằng positional embedding 2D + self-attention — nhìn tất cả patch cùng lúc, tự nhiên hơn cho ảnh.

**Random masking:** LSTM không xử lý được bài toán "cho vài patch bất kỳ, predict phần còn lại" do bản chất sequential. **MAE (Masked Autoencoder)** dùng transformer encoder nhìn toàn bộ patch đã mask, phù hợp hơn cho inpainting.

| | RNN | LSTM | Transformer |
| -- | ----- | ------ | ------------- |
| Xử lý vanishing gradient | Không | Có (cell state) | Không cần (attention trực tiếp) |
| Xử lý song song | Không | Không | Có |
| Bộ nhớ dài hạn | Kém | Tốt | Rất tốt (attention toàn cục) |
| Phù hợp random masking | Không | Không | Có |
| Tham số (cùng capacity) | Ít | Nhiều hơn RNN | Nhiều nhất |

---

[← README chính](../README.md)
