# Deep Learning Practice

Repo này dùng để thực hành các kiến trúc deep learning cơ bản từ đầu bằng PyTorch.

Mỗi mô hình được implement, huấn luyện và trực quan hóa trên bộ dữ liệu MNIST hoặc CIFAR-10.

Mục tiêu là hiểu rõ lý thuyết, toán học và cách vận hành thực tế của từng kiến trúc.

---

## Cấu trúc repo

```text
dl/
├── mnist/
│   ├── mlp_mnist.py       # MLP trên MNIST
│   ├── cnn_mnist.py       # LeNet-5 trên MNIST
│   ├── ae_mnist.py        # Autoencoder trên MNIST
│   ├── vae_mnist.py       # Variational Autoencoder trên MNIST
│   ├── lstm_mnist.py      # Long Short-Term Memory trên MNIST
│   └── vit_mnist.py       # Vision Transformer trên MNIST
├── cifar10/
│   ├── mlp_cifar10.py     # MLP trên CIFAR-10
│   └── cnn_cifar10.py     # CNN trên CIFAR-10
├── docs/
│   ├── mlp.md             # Lý thuyết & thực hành MLP
│   ├── cnn.md             # Lý thuyết & thực hành CNN
│   ├── ae.md              # Lý thuyết & thực hành AE
│   ├── vae.md             # Lý thuyết & thực hành VAE
│   ├── lstm.md            # Lý thuyết & thực hành LSTM
│   └── vit.md             # Lý thuyết & thực hành Transformer / ViT
└── requirements.txt
```

---

## Các chủ đề

| Chủ đề | Tài liệu | Code MNIST | Code CIFAR-10 |
| -------- | ---------- | ------------ | --------------- |
| Multi-Layer Perceptron (MLP) | [docs/mlp.md](docs/mlp.md) | [mlp_mnist.py](mnist/mlp_mnist.py) | [mlp_cifar10.py](cifar10/mlp_cifar10.py) |
| Convolutional Neural Network (CNN) | [docs/cnn.md](docs/cnn.md) | [cnn_mnist.py](mnist/cnn_mnist.py) | [cnn_cifar10.py](cifar10/cnn_cifar10.py) |
| Autoencoder (AE) | [docs/ae.md](docs/ae.md) | [ae_mnist.py](mnist/ae_mnist.py) | — |
| Variational Autoencoder (VAE) | [docs/vae.md](docs/vae.md) | [vae_mnist.py](mnist/vae_mnist.py) | — |
| Long Short-Term Memory (LSTM) | [docs/lstm.md](docs/lstm.md) | [lstm_mnist.py](mnist/lstm_mnist.py) | — |
| Vision Transformer (ViT) | [docs/vit.md](docs/vit.md) | [vit_mnist.py](mnist/vit_mnist.py) | — |

---

## Môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Thiết bị: Apple MPS (M1/M2/M3) → CUDA → CPU (tự động chọn).
