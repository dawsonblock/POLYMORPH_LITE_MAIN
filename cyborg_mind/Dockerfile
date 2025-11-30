# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy only necessary code
COPY . .

# Default entrypoint
CMD ["python", "cyborg_mind/scripts/train_gym_cartpole.py"]
