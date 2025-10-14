# Image-to-Translation Project

Dự án dịch text từ ảnh tiếng Anh sang tiếng Việt sử dụng Vision Transformer và Transformer Decoder.

## Kiến trúc

- **Vision Encoder**: Vision Transformer (ViT) để extract features từ ảnh chứa text
- **Text Decoder**: Transformer decoder với cross-attention để generate text tiếng Việt
- **Cross-Attention**: Kết nối giữa vision features và text generation

## Cài đặt

```bash
# Install requirements
pip install torch torchvision torchtext
pip install underthesea  # Vietnamese tokenizer
pip install spacy  # English tokenizer (optional)
pip install pillow  # Image processing
pip install pandas  # Data handling

# Test installation
python test_project.py
```

## Chuẩn bị dữ liệu

Dữ liệu cần có format JSON:

```json
[
  {
    "id": "00001",
    "image_path": "data/train/images/00001.png",
    "source_text": "Hello World",
    "target_text": "Xin chào thế giới"
  },
  {
    "id": "00002", 
    "image_path": "data/train/images/00002.png",
    "source_text": "Good morning",
    "target_text": "Chào buổi sáng"
  }
]
```

**Cấu trúc thư mục:**
```
data/
├── train/
│   ├── train.json
│   └── images/
│       ├── 00001.png
│       └── 00002.png
└── val/
    ├── val.json
    └── images/
        ├── 10001.png
        └── 10002.png
```

## Training

### Option 1: Normal Training (Requires powerful GPU)
```bash
python train.py -d /path/to/data -m /path/to/model_output
```

### Option 2: Incremental Training (For limited GPU memory)
```bash
# Check your system and optimize config first
python check_memory.py

# Automatic incremental training (recommended)
python incremental_train.py -d /path/to/data -m /path/to/model_output

# Manual incremental training with custom settings
python incremental_train.py -d /path/to/data -m /path/to/model_output -e 3 -mem 4
```

### Option 3: Resume Training
```bash
# Resume from latest checkpoint
python train.py -d /path/to/data -m /path/to/model_output -r latest

# Resume from specific checkpoint
python train.py -d /path/to/data -m /path/to/model_output -r /path/to/checkpoint.pt
```

### Memory Optimization Features

🧠 **Automatic Memory Management:**
- **Mixed Precision Training**: Reduces memory usage by ~40%
- **Gradient Accumulation**: Simulate large batch size with smaller micro-batches
- **Checkpoint Management**: Auto-save and resume training
- **Memory Cleanup**: Automatic GPU memory cleanup between sessions

📊 **Config Auto-Optimization:**
- **2-4GB GPU**: `d_model=256, batch_size=2, micro_batch=1`
- **4-8GB GPU**: `d_model=384, batch_size=4, micro_batch=2`  
- **8-12GB GPU**: `d_model=512, batch_size=6, micro_batch=3`
- **12GB+ GPU**: `d_model=512, batch_size=8, micro_batch=4`

**Tham số config trong `config/config.json`:**
- `d_model`: 512/384/256 - Dimension của transformer (tùy GPU memory)
- `batch_size`: 8/4/2 - Effective batch size 
- `micro_batch_size`: 4/2/1 - Actual batch size per forward pass
- `gradient_accumulation_steps`: 2/4 - Steps to accumulate gradients
- `max_strlen`: 128 - Độ dài tối đa của câu output
- `epoch`: 50 - Số epoch training
- `n_layers`: 6/4 - Số layer của transformer
- `heads`: 8/4 - Số attention heads
- `image_size`: 224 - Kích thước ảnh input
- `mixed_precision`: true - Sử dụng FP16 để tiết kiệm memory
- `checkpoint_every_n_steps`: 500 - Tự động save checkpoint mỗi N steps

## Inference

```bash
python inference.py -p /path/to/image.jpg -m /path/to/trained_model
```

## Cấu trúc dự án

```
├── config/
│   └── config.json          # Cấu hình model với memory optimization
├── modules/
│   ├── transformer.py       # Main model với cross-attention
│   ├── encoder.py           # Vision Transformer encoder
│   ├── decoder.py           # Text decoder
│   └── ...                  # Các module khác
├── utils/
│   ├── dataset.py           # Data loading và preprocessing
│   ├── translator.py        # Inference pipeline
│   ├── beam_search.py       # Beam search algorithm
│   ├── tokenizer.py         # Vietnamese tokenizer
│   ├── image_processor.py   # Image preprocessing
│   ├── checkpoint_manager.py # Training checkpoint management
│   └── ...                  # Các utility khác
├── train.py                 # Training script với resume capability
├── incremental_train.py     # Incremental training for limited memory
├── check_memory.py          # Memory checker và config optimizer
├── inference.py             # Inference script
└── test_project.py          # Test project setup
```

## Lưu ý quan trọng

### Về dữ liệu:
- Ảnh nên có text rõ ràng, không bị mờ hoặc xoay
- Text trong ảnh nên đơn giản, không quá phức tạp
- Bản dịch tiếng Việt cần chính xác và tự nhiên

### Về training:
- **GPU yếu (2-6GB)**: Dùng `incremental_train.py` để train từng phần
- **GPU mạnh (8GB+)**: Có thể dùng `train.py` bình thường
- Batch size có thể cần giảm nếu thiếu memory
- Training time: 10-50 epochs tùy vào kích thước dataset
- **Checkpoint**: Tự động save mỗi 500 steps, có thể resume bất cứ lúc nào

### Về inference:
- Model load một lần, có thể dịch nhiều ảnh
- Beam search với k=5 cho kết quả tốt
- Output sẽ được post-process để clean punctuation

## Ví dụ sử dụng

```python
# Test model
from utils.translator import translate
from utils.image_processor import ImageProcessor
import torch

# Load model (see inference.py for full example)
translation = translate(
    image_path="test_image.jpg", 
    model=model,
    vocab=vocab,
    max_strlen=128,
    device=device,
    k=5,
    image_processor=image_processor
)

print(f"Translation: {translation}")
```

## Troubleshooting

1. **Lỗi tokenizer**: Cài đặt `pip install underthesea`
2. **Lỗi CUDA/Memory**: 
   - Chạy `python check_memory.py` để check GPU
   - Dùng `python incremental_train.py` cho GPU yếu
   - Giảm `batch_size` và `d_model` trong config
3. **Training bị gián đoạn**: 
   - Resume với `python train.py -r latest`
   - Hoặc dùng `incremental_train.py` để tự động resume
4. **Lỗi image loading**: Kiểm tra đường dẫn ảnh trong JSON
5. **Kết quả dịch kém**: Tăng epoch training hoặc cải thiện dataset
6. **Out of Memory**: 
   - Giảm `micro_batch_size` trong config
   - Tăng `gradient_accumulation_steps`
   - Bật `mixed_precision: true`
