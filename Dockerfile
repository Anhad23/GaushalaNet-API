# Dockerfile — use Python 3.11 to match your training env
FROM python:3.11.9-slim

WORKDIR /app

# system deps useful for Pillow/OpenCV and TF on CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copy and install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app code & model (or ensure model path)
COPY . /app

ENV PORT=10000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
