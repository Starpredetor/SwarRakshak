# GPU image. For a CPU-only box swap the base for python:3.11-slim and install
# the CPU torch wheel instead.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3-pip libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 \
 && pip3 install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY scripts ./scripts
COPY config.yaml .

# Bake the weights in rather than fetching at container start -- a demo box
# should not depend on the venue's wifi.
RUN python3 scripts/fetch_models.py || echo "WARNING: model prefetch failed; fallback will be used"

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
