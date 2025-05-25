# Use Python 3.11 (lighter than 3.12) with slim variant
FROM python:3.11-slim

# Minimize layers and clean up aggressively
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/* /usr/share/doc/* /usr/share/man/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages with memory-optimized settings
RUN pip install --upgrade pip --no-cache-dir && \
    # Install required dependencies first
    pip install --no-cache-dir \
        typing-extensions \
        numpy \
        diskcache && \
    # Install llama-cpp-python with minimal features
    CMAKE_ARGS="-DLLAMA_NATIVE=ON -DLLAMA_STATIC=ON -DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_F16C=OFF -DLLAMA_FMA=OFF" \
    pip install --no-cache-dir llama-cpp-python==0.2.90 && \
    # Install other dependencies
    pip install --no-cache-dir \
        flask==3.0.0 \
        flask-cors==4.0.0 \
        gunicorn==21.2.0 \
        gevent>=1.4 \
        psutil==5.9.6 && \
    # Aggressive cleanup
    pip cache purge && \
    rm -rf ~/.cache/pip && \
    find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy application
COPY app.py .

# Create model directory
RUN mkdir -p /app/models

# Set memory-optimized environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV MALLOC_MMAP_THRESHOLD_=100000
ENV PYTHONMALLOC=malloc
ENV GGML_METAL=0
ENV LLAMA_CUBLAS=0
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Expose port
EXPOSE 8080

# Health check with longer intervals to reduce overhead
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=2 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run with minimal resources
# --worker-connections=10: Limit concurrent connections
# --max-requests=100: Restart worker after 100 requests (prevent memory leaks)
# --max-requests-jitter=10: Add randomness to restarts
# --preload: Preload app to save memory
CMD ["gunicorn", \
     "--worker-class", "gevent", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--worker-connections", "10", \
     "--timeout", "180", \
     "--max-requests", "100", \
     "--max-requests-jitter", "10", \
     "--preload", \
     "--log-level", "info", \
     "app:app"]