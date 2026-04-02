# TinyTorch
tinytorch-project/
├── pyproject.toml            # File quan trọng nhất để cài đặt thư viện
├── README.md                 # Hướng dẫn sử dụng và giới thiệu
├── .gitignore                # Bỏ qua các file rác, __pycache__, .ipynb_checkpoints
├── src/                      # Chứa mã nguồn thực tế của thư viện
│   └── tinytorch/            # Tên gói bạn sẽ 'import'
│       ├── __init__.py       # Giúp Python nhận diện thư mục là một package
│       ├── tensor.py         # Module 01: Lớp Tensor, Strides, Broadcasting
│       ├── autograd.py       # Module 02: Computational Graph, Backward pass
│       ├── nn/               # Module 03: Neural Network layers (Linear, ReLU...)
│       │   ├── __init__.py
│       │   ├── modules.py
│       │   └── functional.py
│       └── optim/            # Module 04: Optimizers (SGD, Adam...)
│           ├── __init__.py
│           └── optimizer.py
├── tests/                    # Các bài kiểm tra để đảm bảo thư viện chạy đúng
│   ├── test_tensor.py
│   └── test_autograd.py
├── examples/                 # Các dự án mẫu sử dụng thư viện này
│   └── linear_regression.py  # Ví dụ: Dùng tinytorch giải bài toán thực tế
└── notebooks/                # Nơi bạn lưu các file .ipynb bài tập của khóa học