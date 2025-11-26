FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose BentoML default port
EXPOSE 3000

# Run BentoML service
CMD ["bentoml", "serve", "service:svc", "--host", "0.0.0.0", "--port", "3000"]
