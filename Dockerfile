FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies (onnxruntime explicitly for rembg)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnxruntime

# Copy application code
COPY . ./api/

# Create directories
RUN mkdir -p uploads outputs

# Expose port
EXPOSE 8000

# Default command - run FastAPI with uvicorn (PORT set by Railway)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
