# ── cl_ml Dockerfile ──
# ML pipeline for CarbonLens (training + inference)

FROM python:3.12-slim AS base

WORKDIR /app

# System deps for scientific computing
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python deps (cached layer — this is heavy, keep it separate)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Default: run training. Override CMD for inference.
CMD ["python", "-m", "pipeline.training.train", "--data-dir", "/data", "--output-dir", "/artifacts"]
