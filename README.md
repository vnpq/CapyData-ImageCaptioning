# VN Sport Image Captioning
---

## Overview

Ứng dụng sinh mô tả (caption) tiếng Việt cho ảnh thể thao sử dụng các mô hình **CNN-LSTM**, **CNN-T5** và **ViT-T5**.  
Hệ thống gồm **API backend (FastAPI)** và **frontend HTML/JS** chạy trên trình duyệt, gọi API để nhận caption.

Giao diện web cho phép bạn:

- Upload ảnh hoặc kéo thả trực tiếp vào khung.

- Chọn model từ danh sách (cnn_lstm, cnn_t5, vit_t5).

- Điều chỉnh số lượng caption (1–5).

- Gửi yêu cầu đến API backend và hiển thị kết quả.

---

## Cấu trúc thư mục

IMG-CAPTIONING/
│
├── app_api.py # FastAPI backend
├── model_adapters.py # Adapter load các model
├── my_models/ # Các kiến trúc model CNN-LSTM, CNN-T5, ViT-T5
│ ├── __init__.py
│ ├── cnn_lstm.py
│ ├── cnn_t5.py
│ └── vit_t5.py
│
├── checkpoints/ 
│
├── frontend/ 
│ ├── index.html
│ ├── style.css
│ └── script.js
│
├── requirements.txt 
└── README.md 


---

##  Yêu cầu hệ thống

- Python **3.9+**
- pip
- GPU CUDA (tùy chọn)

---

## Cài đặt

1. **Clone dự án**

```bash
git clone https://github.com/vnpq/CapyData-ImageCaptioning.git
cd CapyData-ImageCaptioning
```

2. **Tạo môi trường ảo & kích hoạt**

``` shell
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\\Scripts\\activate       # Windows
```

Hoặc:

```bash
conda create -n img_captioning_env python=3.9
conda activate img_captioning_env
```

3. **Cài đặt dependencies & kích hoạt môi trường**

```bash
conda env create -f environment.yml
conda activate img_captioning_env
```

4. **Khởi động Backend (API) cục bộ**
Từ thư mục gốc dự án, chạy:
```bash
uvicorn app_api:app --host 0.0.0.0 --port 8000 --reload
```

Hoặc cũng có thể  khởi động bằng cách thực thi tệp tin app_api.py:
```bash
python app_api.py
```

5. **Chạy Front-end**
Mở file frontend/index.html bằng Live Server (VSCode) hoặc bất kỳ HTTP server nào:

```bash
cd frontend
python -m http.server 8080
```

Sau đó truy cập: http://127.0.0.1:8080 
Frontend sẽ gọi API POST /caption ở backend (localhost:8000) để sinh caption.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
## Contributors
| Student ID | Name                   |
|------------|------------------------|
| 22127234   | Cao Hoàng Lộc          |
| 22127360   | Võ Nguyễn Phương Quỳnh |
| 22127450   | Phạm Anh Văn           |