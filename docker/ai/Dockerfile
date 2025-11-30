# AI Service Dockerfile
# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
# Install CPU-only torch to save space (~700MB vs 2GB+)
RUN pip install --user --no-cache-dir scikit-learn joblib
RUN pip install --user --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Stage 2: Runner
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 5000

# Simple Flask wrapper or BentoML serve
CMD ["python", "ai/service_runner.py"] 
