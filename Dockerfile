FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app
ENV CKPT_DIR=/app/checkpoints

EXPOSE 8000
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]