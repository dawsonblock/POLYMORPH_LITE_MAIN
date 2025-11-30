#!/bin/bash
set -e

echo "🚀 Starting Optimized Build..."

# 1. Build API
echo "📦 Building API..."
docker build -t polymorph-api:latest -f docker/api/Dockerfile .

# 2. Build UI
echo "📦 Building UI..."
docker build -t polymorph-ui:latest -f docker/ui/Dockerfile .

# 3. Build AI
echo "📦 Building AI Service..."
docker build -t polymorph-ai:latest -f docker/ai/Dockerfile .

echo "✅ Build Complete!"
echo "-----------------------------------"
echo "Image Sizes:"
docker images | grep polymorph
