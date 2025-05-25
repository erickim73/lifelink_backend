# LifeLink Backend

<!-- Added logo section matching the main README structure -->
<div align="center">
  <img src="./public/lifelink_logo.png" alt="LifeLink Logo" height="60">
  <img src="./public/lifelink.svg" alt="LifeLink" height="60" style="margin-left: 20px;">
  
  **AI-Powered Health API Backend**
  
  [Website Link](https://lifelink-app.vercel.app/) | [Frontend Repo](https://github.com/erickim73/lifelink_frontend/tree/master) 
</div>

---

## Overview

The LifeLink backend is a Flask-based API server that powers the AI health consultation features. It runs a quantized Mistral-7B model optimized for medical question answering, with aggressive memory management designed for resource-constrained environments like AWS EC2 t4g.small instances.

## Architecture

**Framework**: Flask with CORS support for cross-origin requests  
**AI Model**: Mistral-7B-Instruct-v0.3 with IQ1_S quantization  
**Model Runner**: llama.cpp for efficient inference  
**Deployment**: Docker container with Gunicorn WSGI server  
**Memory Management**: Adaptive model loading/unloading with monitoring  

## Key Features

**Memory-Optimized Model Loading**  
Lazy loading with automatic unloading after 3 minutes of inactivity

**Streaming Response Support**  
Server-sent events for real-time chat interface updates

**Health Monitoring**  
Built-in health check endpoint with memory usage reporting

**Graceful Resource Management**  
Automatic cleanup and memory pressure monitoring

**CORS Configuration**  
Pre-configured for frontend integration with multiple origins

## API Endpoints

### Health Check
```http
GET /health
```
Returns system status and memory usage information.

**Response:**
```json
{
  "status": "healthy",
  "memory_usage_percent": "65.2%",
  "memory_usage_mb": "1024.5MB",
  "model_loaded": true
}
```

### Chat Stream
```http
POST /chat/stream
```
Processes health queries with streaming responses.

**Request Body:**
```json
{
  "newPrompt": "I have a headache and feel tired",
  "userProfile": {
    "first_name": "John",
    "dob": "1990-01-01",
    "gender": "male",
    "medical_conditions": "hypertension",
    "medications": "lisinopril"
  }
}
```

**Response:**
Server-sent events stream with medical advice chunks.

## Installation & Setup

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)
- 2GB+ available RAM
- Model file: `Mistral-7B-Instruct-v0.3.IQ1_S.gguf`

### Model Download
Download the GGUF quantized model files from HuggingFace:
- **Model Repository**: [Mistral-7B-Instruct-v0.3-GGUF](https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/tree/main)
- **Recommended for 2GB RAM**: `Mistral-7B-Instruct-v0.3.IQ1_S.gguf` (smallest quantization)
- **For 4GB+ RAM**: `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` (better quality)
- **For 8GB+ RAM**: `Mistral-7B-Instruct-v0.3.Q8_0.gguf` (highest quality)

### Local Development

```bash
# Clone and navigate to backend directory
cd lifelink-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set model path (optional - defaults to current directory)
export MODEL_PATH="./Mistral-7B-Instruct-v0.3.IQ1_S.gguf"

# Run development server
python app.py
```

### Docker Deployment

```bash
# Build container
docker build -t lifelink-backend .

# Run container with model volume
docker run -d \
  -p 8080:8080 \
  -v /path/to/model:/app/models \
  -e MODEL_PATH="/app/models/Mistral-7B-Instruct-v0.3.IQ1_S.gguf" \
  --name lifelink-backend \
  lifelink-backend
```

### AWS EC2 Deployment

```bash
# For t4g.small instances (2 vCPUs, 2GB RAM)
docker run -d \
  -p 8080:8080 \
  -v /home/ec2-user/models:/app/models \
  -e MODEL_PATH="/app/models/Mistral-7B-Instruct-v0.3.IQ1_S.gguf" \
  --memory="1800m" \
  --memory-swap="2000m" \
  --restart unless-stopped \
  --name lifelink-backend \
  lifelink-backend
```

## Memory Optimization

### Model Configuration
The application uses ultra-minimal model settings for resource efficiency:

- **Context Window**: 256 tokens (drastically reduced)
- **Batch Size**: 8 tokens (minimal processing)
- **Threads**: 2 (matches t4g.small vCPUs)
- **Memory Mapping**: Enabled for efficient file access
- **Key-Value Cache**: 16-bit precision to save memory

### Automatic Management
- **Inactivity Timeout**: Model unloads after 3 minutes
- **Memory Pressure**: Automatic cleanup at 80% memory usage
- **Garbage Collection**: Aggressive cleanup after each request
- **Memory Monitoring**: Background thread monitors system resources

## Dependencies

```python
# Core web framework
flask==3.0.0
flask-cors==4.0.0

# WSGI server and async support
gunicorn==21.2.0
gevent==23.9.1

# AI model inference
llama-cpp-python==0.2.90

# System monitoring and utilities
psutil==5.9.6
typing-extensions
numpy
diskcache
```

## Environment Variables

```bash
# Model file path
MODEL_PATH="./Mistral-7B-Instruct-v0.3.IQ1_S.gguf"

# Memory optimization settings
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
MALLOC_TRIM_THRESHOLD_=100000
OMP_NUM_THREADS=1

# Disable GPU acceleration (CPU-only)
GGML_METAL=0
LLAMA_CUBLAS=0
```

## Docker Configuration

The Dockerfile includes several optimizations:

**Base Image**: Python 3.11-slim for reduced size  
**Build Tools**: Minimal GCC/CMake installation with cleanup  
**Model Compilation**: CPU-optimized llama.cpp build  
**Memory Settings**: Environment variables for malloc optimization  
**Health Checks**: Automatic container health monitoring  

## Security Considerations

**CORS Origins**: Restricted to specific frontend domains  
**Input Validation**: Request payload validation and sanitization  
**Model Isolation**: Sandboxed model execution environment  
**Resource Limits**: Memory and CPU constraints via Docker  
**Error Handling**: Graceful degradation without exposing internals  

## Monitoring & Debugging

### Health Check Monitoring
```bash
# Check system status
curl http://localhost:8080/health

# Monitor logs
docker logs -f lifelink-backend
```

### Memory Usage Tracking
The application logs memory usage at key points:
- Model loading/unloading events
- High memory pressure warnings
- Automatic cleanup triggers

### Performance Tuning
For different instance sizes, adjust these parameters in `app.py`:

```python
# Commented configuration examples for different setups
# t4g.small (2GB): Current settings
n_ctx=256, n_batch=8, max_tokens=150

# t4g.medium (4GB): Increased capacity  
# n_ctx=512, n_batch=16, max_tokens=300

# t4g.large (8GB): Full performance
# n_ctx=1024, n_batch=32, max_tokens=500
```

## Troubleshooting

**Model Loading Fails**  
- Check available memory (need 600MB+ free)
- Verify model file path and permissions
- Ensure sufficient disk space

**High Memory Usage**  
- Monitor `/health` endpoint regularly
- Check for memory leaks in application logs
- Restart container if memory doesn't decrease

**Slow Response Times**  
- Reduce `max_tokens` in model configuration
- Check CPU utilization and system load
- Consider upgrading to larger instance type


---

**Optimized for reliable AI health consultations on resource-constrained infrastructure**