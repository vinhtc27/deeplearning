# Transformer & Vision Transformer (ViT)

[← README chính](../README.md)

---

## 1. Lý thuyết

**Vấn đề của LSTM:** LSTM giải quyết vanishing gradient nhưng vẫn còn hai hạn chế — sequential dependency (token $t$ phải đợi $t-1$ xử lý xong, không thể song song hóa) và bottleneck bộ nhớ (thông tin từ token xa phải nén qua nhiều gate).

Transformer giải quyết bằng **self-attention**: mỗi token nhìn trực tiếp vào tất cả token khác trong một bước tính toán, không phụ thuộc khoảng cách vị trí. Ví dụ trong câu *"The animal didn't cross the street because **it** was too tired"*: self-attention tính được "it" chú ý vào "animal" hơn "street" mà không cần lan truyền qua nhiều bước thời gian. Ba thành phần cốt lõi: **Multi-Head Self-Attention** (nhiều head song song, mỗi head học quan hệ khác nhau), **Position-wise FFN** (MLP nhỏ xử lý mỗi token độc lập sau attention), **Positional Encoding** (inject thông tin vị trí vì attention không phân biệt thứ tự).

**Transformer gốc (Vaswani et al., 2017)** thiết kế cho NLP: input là text token IDs → embedding lookup → Encoder-Decoder. Decoder dùng **causal mask** để không nhìn trước token tương lai khi generate — attention score tại vị trí $j > i$ bị set $-\infty$ trước softmax. Positional encoding sinusoidal cố định.

**Vision Transformer (Dosovitskiy et al., 2020)** adapt Transformer cho ảnh. Ảnh không có token tự nhiên; pixel đơn lẻ quá nhiều (28×28 = 784 token → attention matrix 784×784, rất nặng). Giải pháp: chia ảnh thành $N$ patch, flatten và chiếu tuyến tính lên $d_\text{model}$ — mỗi patch thành một token. Sau đó chạy đúng Transformer encoder, không thay đổi gì về toán học attention.

```text
Transformer gốc (NLP):
  text → tokenizer → embedding lookup (vocab × d_model) → Transformer Encoder-Decoder

ViT (Vision):
  image → patch split → flatten → Linear(patch_size² × C → d_model) → Transformer Encoder
```

Các điểm ViT khác Transformer gốc:

| | Transformer gốc | ViT |
| -- | --------------- | --- |
| **Input** | Discrete token IDs → embedding lookup | Liên tục: pixel values → linear projection |
| **Kiến trúc** | Encoder + Decoder | Encoder only (task là classification) |
| **Positional encoding** | Sinusoidal cố định | Learnable (ảnh kích thước cố định) |
| **Token đặc biệt** | `<BOS>`, `<EOS>`, `<PAD>` | `[CLS]` — aggregate toàn bộ ảnh để classify |
| **Output** | Chuỗi token mới (seq2seq) | Vector của `[CLS]` → classifier head |

**Điểm mấu chốt:** ViT không phát minh gì mới về attention — chứng minh rằng patch embedding đủ để "tokenize" ảnh, sau đó Transformer encoder tiêu chuẩn xử lý tốt hơn CNN khi có đủ dữ liệu.

Các biến thể kiến trúc chính:

| Biến thể | Cấu trúc | Ứng dụng |
| -- | -- | -- |
| **Encoder-only** (BERT) | Stack encoder blocks | Classification, NER, Q&A |
| **Decoder-only** (GPT) | Stack decoder blocks (causal mask) | Language generation, LLM |
| **Encoder-Decoder** (T5, original Transformer) | Encoder + Decoder + cross-attention | Translation, summarization |
| **Vision Transformer (ViT)** | Encoder-only trên image patches | Image classification |

---

## 2. Toán học

### Scaled Dot-Product Attention

Cho một chuỗi $n$ token, mỗi token được chiếu thành ba vector:

- **Query** $Q \in \mathbb{R}^{n \times d_k}$: "token này đang tìm gì?"
- **Key** $K \in \mathbb{R}^{n \times d_k}$: "token này có thể cung cấp gì?"
- **Value** $V \in \mathbb{R}^{n \times d_v}$: "nội dung thực sự của token này"

Attention score giữa token $i$ và $j$ là tích vô hướng $q_i \cdot k_j$, sau đó scale và softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Tại sao chia $\sqrt{d_k}$?** Tích vô hướng $q \cdot k$ có phương sai $d_k$ (nếu các thành phần i.i.d. với var = 1). Khi $d_k$ lớn, giá trị lớn → softmax bão hòa về 0 hoặc 1 → gradient gần bằng 0. Chia $\sqrt{d_k}$ đưa phương sai về 1, giữ softmax trong vùng gradient ổn định.

**Ma trận attention** $A = \text{softmax}(QK^\top / \sqrt{d_k}) \in \mathbb{R}^{n \times n}$: hàng $i$ là phân phối xác suất — token $i$ chú ý bao nhiêu vào mỗi token trong chuỗi. Output là trung bình có trọng số của Value:

$$\text{output}_i = \sum_{j=1}^{n} A_{ij} \cdot v_j$$

**Complexity:** $O(n^2 d_k)$ — bình phương theo chiều dài chuỗi, đây là bottleneck của Transformer với chuỗi rất dài.

### Causal Masking (Decoder)

Để mô hình generation không "nhìn trước" token tương lai, ma trận attention bị mask:

$$A_{ij} = -\infty \quad \text{nếu } j > i$$

Sau softmax, $A_{ij} = 0$ với $j > i$ → token $i$ chỉ attend vào token $\leq i$.

### Multi-Head Attention

Thay vì một attention function với $d_\text{model}$ chiều, chia thành $h$ head, mỗi head dùng $d_k = d_v = d_\text{model} / h$:

$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

Trong đó $W_i^Q, W_i^K \in \mathbb{R}^{d_\text{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_\text{model} \times d_v}$, $W^O \in \mathbb{R}^{h d_v \times d_\text{model}}$.

**Tại sao multi-head?** Mỗi head có thể học một kiểu quan hệ khác nhau trong không gian con riêng. Ví dụ head 1 học quan hệ cú pháp subject-verb, head 2 học co-reference, head 3 học proximity... Một head duy nhất bị trung bình hóa qua tất cả các quan hệ này.

**Số tham số attention:** $4 \times d_\text{model}^2$ (3 projection Q/K/V + output projection $W^O$).

### Positional Encoding

Attention không phân biệt thứ tự token — hoán vị input cho cùng kết quả (ngoại trừ output cũng bị hoán vị theo). Cần inject thông tin vị trí vào embedding.

**Sinusoidal Positional Encoding** (Vaswani et al.):

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

- $pos$: vị trí token trong chuỗi ($0, 1, \ldots, n-1$)
- $i$: chỉ số chiều embedding ($0, 1, \ldots, d_\text{model}/2 - 1$)

Mỗi chiều tương ứng một sóng sin/cos với tần số khác nhau — từ rất chậm ($i=0$, bước sóng $2\pi \times 10000$) đến rất nhanh ($i = d_\text{model}/2 - 1$, bước sóng $\approx 2\pi$). Tổ hợp tạo ra "chữ ký nhị phân" duy nhất cho mỗi vị trí.

**Tại sao sin/cos?** $PE_{pos+k}$ có thể biểu diễn tuyến tính từ $PE_{pos}$ — mô hình có thể học quan hệ vị trí tương đối dễ hơn.

Embedding đầu vào = token embedding + positional encoding:

$$x'_{pos} = x_{pos} + PE_{pos}$$

### Feed-Forward Network (FFN)

Sau attention, mỗi token được xử lý **độc lập** qua một MLP 2 lớp:

$$\text{FFN}(x) = \text{Act}(xW_1 + b_1)\, W_2 + b_2$$

- $W_1 \in \mathbb{R}^{d_\text{model} \times d_{ff}}$, thường $d_{ff} = 4 \times d_\text{model}$
- $W_2 \in \mathbb{R}^{d_{ff} \times d_\text{model}}$
- Activation: ReLU hoặc GELU (Transformer thường dùng GELU vì smooth hơn ReLU tại 0)

**Tác dụng — phân công rõ với attention:**

| Sub-layer | Làm gì | Token tương tác? |
| --------- | ------- | ---------------- |
| Attention | Tổng hợp thông tin từ các token khác | Có (cross-token) |
| FFN | Xử lý và biến đổi từng token | Không (per-token) |

Attention giải quyết câu hỏi *"token này cần thông tin từ đâu?"* — FFN giải quyết câu hỏi *"làm gì với thông tin đó?"*. Thiếu FFN, mô hình chỉ tính weighted average của value vectors — rất hạn chế về khả năng biểu diễn phi tuyến.

**Tại sao expand rồi contract ($d_{ff} = 4 d_\text{model}$)?** Expand lên không gian cao chiều → nonlinearity cắt bớt → project về $d_\text{model}$. Pattern này tạo ra "working space" lớn để mô hình thực hiện tính toán phức tạp hơn trước khi nén lại. Lý do $\times 4$ là empirical — hoạt động tốt trong thực tế.

**Chi phí tham số:** FFN chiếm phần lớn tham số trong mỗi block — $2 \times d_\text{model} \times d_{ff} = 8 \times d_\text{model}^2$, so với $4 \times d_\text{model}^2$ của attention.

### Layer Normalization & Residual Connection

#### Residual Connection

Mỗi sub-layer bọc trong residual shortcut:

$$\text{output} = x + \text{Sublayer}(x)$$

**Vấn đề với mạng sâu không có residual:** gradient backprop qua $L$ layer liên tiếp bị nhân chuỗi Jacobian — nếu mỗi layer co gradient xuống một chút, tích $L$ layer → vanish. VGG-19 (không residual) khó train hơn ResNet-50 (có residual) dù ít layer hơn.

**Tại sao residual giải quyết được:** đạo hàm loss theo $x$:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial \text{output}} \cdot \left(1 + \frac{\partial \text{Sublayer}(x)}{\partial x}\right)$$

Số $1$ đảm bảo gradient luôn có đường chảy thẳng về — dù $\frac{\partial \text{Sublayer}}{\partial x}$ nhỏ đến đâu, gradient không vanish hoàn toàn. Sublayer chỉ cần học **phần dư** (residual) so với identity — dễ hơn học full transformation từ đầu.

#### Layer Normalization

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

$\mu$ và $\sigma^2$ tính trên **chiều feature** $d_\text{model}$ của từng token riêng lẻ:

$$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$$

$\gamma, \beta \in \mathbb{R}^d$ — learnable scale và shift, cho phép mạng undo normalization nếu cần.

**Tác dụng:** giữ activation trong vùng có gradient ổn định — tránh exploding/vanishing activation qua nhiều layer. Không cần learning rate nhỏ hay khởi tạo cẩn thận.

**LayerNorm vs BatchNorm:**

| | BatchNorm | LayerNorm |
| - | --------- | --------- |
| Normalize theo | Batch (axis 0) | Feature (axis -1) |
| $\mu$, $\sigma^2$ phụ thuộc | Cả batch | Từng sample độc lập |
| Vấn đề | Cần batch đủ lớn; không ổn với seq dài khác nhau; inference với batch=1 khác train | Không có |
| Dùng cho | CNN, batch lớn | Transformer, RNN |

BatchNorm tính thống kê qua batch → cần batch đủ lớn để ổn định, và mean/var khác nhau giữa train/inference. LayerNorm tính trên mỗi token độc lập → không phụ thuộc batch size, hoạt động nhất quán cả train lẫn inference.

#### Pre-LN vs Post-LN

**Post-LN** (paper gốc Vaswani 2017):
$$x' = \text{LayerNorm}(x + \text{Attention}(x))$$
$$x'' = \text{LayerNorm}(x' + \text{FFN}(x'))$$

**Pre-LN** (hầu hết impl hiện đại, kể cả repo này):
$$x' = x + \text{Attention}(\text{LayerNorm}(x))$$
$$x'' = x' + \text{FFN}(\text{LayerNorm}(x'))$$

**Tại sao Pre-LN ổn định hơn:** với Post-LN, gradient phải đi qua LN trên đường về — LN có thể scale gradient lên hoặc xuống không kiểm soát ở giai đoạn đầu train. Với Pre-LN, residual path $x \to x'$ hoàn toàn sạch, gradient chảy thẳng không qua LN → ổn định hơn, có thể train không cần warmup.

### Encoder Block đầy đủ

Một encoder block gồm hai sub-layer:

```text
x → LayerNorm → MultiHead Self-Attention → + x → LayerNorm → FFN → + x
     ↑__________________________________|        ↑________________________|
            residual                                     residual
```

Toán học (Pre-LN):

$$x' = x + \text{MultiHead}(\text{LN}(x),\; \text{LN}(x),\; \text{LN}(x))$$
$$x'' = x' + \text{FFN}(\text{LN}(x'))$$

Stack $N$ encoder block → Transformer Encoder.

### Patch Embedding — Tokenize ảnh

Transformer gốc nhận discrete token IDs (word index). Ảnh là tensor liên tục — cần cách chuyển sang token. ViT dùng **patch embedding**:

1. Cắt ảnh $H \times W$ thành $N$ patch kích thước $P \times P$: $N = \frac{HW}{P^2}$
2. Flatten mỗi patch: $P^2 \times C$ pixel values → vector $\mathbb{R}^{P^2 C}$
3. Chiếu tuyến tính vào $d_\text{model}$: $E = \text{Linear}(P^2 C, d_\text{model})$

$$z_i = E \cdot \text{flatten}(\text{patch}_i) + p_i$$

Kết quả: $N$ token embedding — đúng format input của Transformer encoder.

**Tại sao không dùng từng pixel?** $28 \times 28 = 784$ token → attention matrix $784 \times 784$ = 615K entries. Với patch $7 \times 7$: chỉ $16$ token → $16 \times 16 = 256$ entries, rẻ hơn 2400×.

### CLS Token và Classification Head

Transformer encoder cho ra $N$ vector — một vector mỗi patch. Để classify toàn bộ ảnh cần **aggregate** thông tin từ tất cả patch thành 1 vector.

**Cách ViT:** thêm một token đặc biệt `[CLS]` không tương ứng patch nào vào đầu sequence:

$$\text{input} = [\text{CLS},\; z_1,\; z_2,\; \ldots,\; z_N] \in \mathbb{R}^{(N+1) \times d_\text{model}}$$

Sau $L$ encoder layer, CLS token đã attend vào tất cả patch qua bidirectional attention → biểu diễn global của ảnh. Đưa vào linear classifier:

$$\hat{y} = \text{Linear}(d_\text{model}, C) \cdot h_{\text{CLS}}$$

**Tại sao CLS attend được toàn bộ?** Encoder dùng bidirectional attention (không có causal mask) — mọi token nhìn tất cả token khác. CLS không bị ràng buộc bởi vị trí không gian → tự do tổng hợp thông tin từ bất kỳ patch nào.

**Cách thay thế:** average pooling tất cả $N$ patch token (không cần CLS) — phổ biến trong các impl hiện đại, đôi khi tốt hơn CLS.

### Cross-Entropy Loss cho phân loại

Với bài toán $C$ lớp, output của mô hình là logit $z \in \mathbb{R}^C$ (lấy từ CLS token):

$$p_c = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}} \quad \text{(softmax)}$$

$$\mathcal{L} = -\log p_{y} = -z_y + \log \sum_{j=1}^C e^{z_j}$$

Trong đó $y$ là class thật. Gradient theo logit:

$$\frac{\partial \mathcal{L}}{\partial z_c} = p_c - \mathbf{1}[c = y]$$

Gradient đơn giản, bounded — lý do cross-entropy phù hợp cho classification.

---

## 3. Kiến trúc trong repo

### MNIST — Image Classification với Vision Transformer (`vit_mnist.py`)

Task: phân loại 10 chữ số MNIST (28×28) bằng Transformer encoder — không dùng convolution.

**Patch flow:**

```text
(B, 1, 28, 28)
  → 16 patch 7×7 (unfold)                    → (B, 16, 49)
  → Linear(49 → 128)  [patch embedding]      → (B, 16, d_model)
  → prepend [CLS] token                      → (B, 17, d_model)
  → + learnable pos_embed (17 vị trí)        → (B, 17, d_model)
  → EncoderBlock × 4                         → (B, 17, d_model)
  → LN → lấy x[:, 0]  [CLS output]          → (B, d_model)
  → Linear(128 → 10)                         → (B, 10) logits
```

**Kiến trúc ViT:**

```text
Hyperparameters:
  d_model  = 128   (embedding dim)
  n_heads  = 4     (số attention head)
  d_k = d_v = 32   (d_model / n_heads)
  d_ff     = 512   (4 × d_model)
  n_layers = 4     (số encoder block)
  dropout  = 0.1

EncoderBlock (× 4):
  Pre-LN → MultiHead Self-Attention(128, 4 heads, bidirectional) → Residual
  Pre-LN → FFN(128 → 512 → 128, GELU)                           → Residual

Head:
  LN(CLS) → Linear(128 → 10) → logits
```

**Số tham số (~802K):**

```text
patch_embed  : 49 × 128 + 128                         =   6,400
CLS token    : 128                                    =     128
pos_embed    : 17 × 128                               =   2,176

Per EncoderBlock:
  qkv (bias=False): 128 × (3×128)                    =  49,152
  proj            : 128 × 128 + 128                  =  16,512
  FFN             : 128×512+512 + 512×128+128        = 131,712
  LayerNorms (×2) : 2 × 256                          =     512
  Total per block                                    = 197,888

4 blocks                                             = 791,552
Final LayerNorm                                      =     256
Classifier: Linear(128×10 + 10)                      =   1,290

Tổng: ~801,802 tham số
```

**Training:**

```text
Loss      : CrossEntropyLoss
Optimizer : AdamW (lr=1e-3, weight_decay=1e-4)
Scheduler : CosineAnnealingLR (T_max=epochs)
Epochs    : 30
Batch     : 256
Dropout   : 0.1
```

**AdamW vs Adam:** Adam + weight decay thông thường apply decay lên adaptive scale → không decay đúng. AdamW tách weight decay ra khỏi gradient update:

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_t$$

Chuẩn hơn về mặt lý thuyết, thường converge tốt hơn với Transformer.

**5 Visualizations:**

| File | Nội dung |
| ---- | -------- |
| `training_curves.png` | Loss + accuracy curves qua các epoch |
| `accuracy_confusion.png` | Per-class accuracy bar + confusion matrix |
| `cls_attention_heatmap.png` | CLS attention overlay lên ảnh gốc — model nhìn vào đâu khi classify |
| `multihead_attention.png` | Attention per head per layer cho 1 ảnh — mỗi head học đặc trưng khác nhau |
| `attention_entropy.png` | Entropy của CLS attention qua các layer per digit class |
| `patch_masking.png` | Mask dần patch theo attention weight — confidence sụp đổ khi xóa patch quan trọng |

---

## 4. Giới hạn và so sánh

**Data-hungry:** Transformer có rất ít inductive bias — không giả định locality hay translation equivariance như CNN. Phải học tất cả từ dữ liệu → cần nhiều data hơn CNN để đạt cùng accuracy trên tập nhỏ. ViT-Base chỉ match CNN khi pretrain trên ImageNet-21k (14M ảnh) trở lên; trên CIFAR-10 hay MNIST, CNN vẫn hiệu quả hơn nếu train from scratch.

**Quadratic attention:** Complexity $O(n^2 \cdot d)$ theo chiều dài chuỗi — với ảnh độ phân giải cao (224×224, patch 16×16 → 196 token) vẫn ổn, nhưng với chuỗi dài hàng nghìn token thì memory là bottleneck. Các biến thể như Longformer, Flash Attention giải quyết vấn đề này.

**Không extrapolate tốt ra độ phân giải mới:** Learnable positional embedding học vị trí cố định (17 vị trí: 1 CLS + 16 patch 7×7 của ảnh 28×28). Nếu test trên ảnh 56×56 (64 patch + 1 CLS), pos embed không match → cần interpolate. Sinusoidal encoding của Transformer gốc tốt hơn về điểm này.

**ViT overkill cho MNIST?** CNN đơn giản đạt >99% accuracy trên MNIST, trong khi ViT cần nhiều tham số và epoch hơn để match. Tuy nhiên, ViT trên MNIST vẫn hữu ích để **học kiến trúc** — CLS attention heatmap visualize được "model nhìn vào đâu", multi-head attention cho thấy mỗi head học đặc trưng không gian khác nhau, patch masking confirm tầm quan trọng thực sự của từng vùng ảnh.

**Complexity so sánh LSTM vs Transformer:**

| | LSTM | Transformer |
| -- | ------ | ------------- |
| **Độ phức tạp mỗi layer** | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| **Số phép tính tuần tự** | $O(n)$ | $O(1)$ |
| **Đường đi tối đa giữa 2 token** | $O(n)$ | $O(1)$ |
| **Song song hóa training** | Không | Có |
| **Phù hợp chuỗi cực dài ($n \gg d$)** | Tốt hơn | Kém hơn ($n^2$ memory) |

**So sánh với các kiến trúc đã học:**

| | MLP | CNN | LSTM | Transformer |
| -- | ----- | ----- | ------ | ------------- |
| **Xử lý không gian 2D** | Flatten (mất cấu trúc) | Tự nhiên (inductive bias) | Tuần tự theo patch | Patch-based, không bias |
| **Xử lý chuỗi** | Không | Không | Tự nhiên | Tự nhiên |
| **Song song hóa** | Có | Có | Không | Có |
| **Quan hệ xa** | Không | Hạn chế (receptive field) | Tốt (cell state) | Rất tốt (direct attention) |
| **Dữ liệu cần để train tốt** | Ít | Vừa | Vừa | Nhiều (ít inductive bias) |
| **Transferability** | Kém | Tốt (conv filter) | Vừa | Rất tốt (pretraining) |

**Tại sao ViT dùng [CLS] token?** Token [CLS] không tương ứng patch nào — thu thập thông tin từ tất cả patch qua attention sau $N$ layer. Biểu diễn [CLS] đại diện toàn bộ ảnh → đưa vào classifier. Cách khác: average pooling tất cả patch token (cũng phổ biến thực tế, đôi khi tốt hơn).

---

[← README chính](../README.md)
