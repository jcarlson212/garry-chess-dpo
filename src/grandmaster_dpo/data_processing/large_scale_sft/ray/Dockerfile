FROM anyscale/ray:2.52.1-slim-py311
# (If available for your Ray version, consider slim for smaller images:
# FROM anyscale/ray:2.52.1-slim-py311-cu118)

# Install system deps
USER root
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends stockfish && \
    rm -rf /var/lib/apt/lists/*
USER ray

# Python deps (match your pyproject, and avoid tkinter via pip)
# If you want CUDA-enabled torch via pip, point to the cu118 index.
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir \
      chess==1.10.0 \
      einops==0.8.0 \
      gdown==5.2.0 \
      numpy==1.26.4 \
      pandas==2.2.3 \
      pyyaml==6.0.2 \
      pyzstd==0.15.9 \
      requests==2.32.3 \
      torch==2.2.2 \
      tqdm==4.65.0 \
      orjson \
      maia2 \
      tensorboard \
      packaging \
      s3fs
