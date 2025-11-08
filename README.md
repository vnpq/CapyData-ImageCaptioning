
# 🇻🇳 VN Sport Image Captioning

---

## 🧠 Overview

**VN Sport Image Captioning** is a Vietnamese-language sports image captioning application powered by **transformer-based ViT-T5 models**.
The system includes both a **FastAPI backend** and a lightweight **HTML/JavaScript frontend**, enabling real-time caption generation directly in the browser.

The web interface allows users to:

* Upload or drag-and-drop a sports image.
* Adjust the number of captions generated (1–5).
* Send requests to the backend API and display captions instantly.

---

## 🗂️ Project Structure

```bash
CapyData-ImageCaptioning/
│
├── api/                    # FastAPI backend
│   └── app.py              
│
├── checkpoints/            # Model checkpoints
│
├── configs/                # Configuration files
│
├── data_notebook/          # Data collection, labeling, preprocessing, and visualization notebooks
│   ├── data_collection.ipynb
│   ├── data_preprocessing.ipynb
│   └── data_exploring.ipynb
│
├── dataset/ 
│   ├── metadata/           # Dataset metadata
│   └── capydata_ic/        # Main dataset storage
│
├── model/                  # Model architecture, training, and evaluation
│   ├── __init__.py
│   ├── evaluate.ipynb
│   ├── train.ipynb
│   └── vit_t5.py
│
├── frontend/               # Frontend web interface
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

## ⚙️ System Requirements

* **Python 3.9+**
* **pip**
* **CUDA-compatible GPU** *(optional, for faster inference)*

---

## 🚀 Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/vnpq/CapyData-ImageCaptioning.git
cd CapyData-ImageCaptioning
```

### 2. Create and activate a virtual environment

#### Using `venv`

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows
```

#### Using Conda

```bash
conda create -n img_captioning_env python=3.9
conda activate img_captioning_env
```

### 3. Install dependencies

```bash
conda env create -f environment.yml
conda activate img_captioning_env
```

### 4. Start the Backend (API)

From the project root:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Launch the Frontend

Open the `frontend` folder and run:

```bash
cd frontend
python -m http.server 8080
```

Then visit:
👉 **[http://0.0.0.0:8080](http://0.0.0.0:8080)**

The frontend will call the backend’s `POST /caption` endpoint (at `localhost:8000`) to generate captions for uploaded images.

---

## 🖼️ Demo

*(Insert demo image below — for example, a sports photo and its generated Vietnamese caption.)*

> **Example:**
>
> ![Demo Image Placeholder](<img width="1858" height="1009" alt="image" src="https://github.com/user-attachments/assets/2adf732d-c1dc-4965-8627-528eedd263f8" />)

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👥 Contributors

| Student ID | Name                   | Role        |
| ---------- | ---------------------- | ----------- |
| 22127450   | **Phạm Anh Văn**       | Team Leader |
| 22127234   | Cao Hoàng Lộc          |             |
| 22127360   | Võ Nguyễn Phương Quỳnh |             |

