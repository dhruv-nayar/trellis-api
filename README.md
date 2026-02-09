# TRELLIS API

REST API for converting images to 3D GLB files using Microsoft's TRELLIS.

## Features

- 🚀 Fast async processing with background jobs
- 📊 Job status tracking
- 🔄 Automatic file cleanup
- 🐳 Docker support
- 📝 OpenAPI documentation
- 🌐 CORS enabled

## Quick Start

### Option 1: Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python main.py

# Or with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t trellis-api .
docker run -p 8000:8000 trellis-api
```

## API Endpoints

### Health Check

```bash
GET /health
```

### Convert Image

```bash
POST /api/convert
Content-Type: multipart/form-data

Parameters:
  - file: Image file (JPG, PNG, WebP)
  - seed: Random seed (optional, default: 1)

Response:
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Image uploaded. Processing will begin shortly."
}
```

### Check Status

```bash
GET /api/status/{job_id}

Response:
{
  "job_id": "uuid",
  "status": "completed",
  "message": "Conversion completed successfully",
  "download_url": "/api/download/uuid"
}
```

### Download Result

```bash
GET /api/download/{job_id}

Returns: GLB file (model/gltf-binary)
```

### Cancel Job

```bash
DELETE /api/jobs/{job_id}

Response:
{
  "message": "Job uuid cancelled and cleaned up"
}
```

## Usage Examples

### cURL

```bash
# Upload image
curl -X POST "http://localhost:8000/api/convert" \
  -F "file=@image.jpg" \
  -F "seed=1"

# Check status
curl "http://localhost:8000/api/status/JOB_ID"

# Download result
curl "http://localhost:8000/api/download/JOB_ID" -o output.glb
```

### Python

```python
import requests

# Upload image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/convert',
        files={'file': f},
        params={'seed': 1}
    )
job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:8000/api/status/{job_id}').json()
print(status)

# Download when ready
if status['status'] == 'completed':
    glb = requests.get(f'http://localhost:8000/api/download/{job_id}')
    with open('output.glb', 'wb') as f:
        f.write(glb.content)
```

### JavaScript

```javascript
// Upload image
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/api/convert?seed=1', {
  method: 'POST',
  body: formData
});
const { job_id } = await response.json();

// Poll for completion
const checkStatus = async () => {
  const status = await fetch(`http://localhost:8000/api/status/${job_id}`);
  const data = await status.json();

  if (data.status === 'completed') {
    // Download result
    const glb = await fetch(`http://localhost:8000/api/download/${job_id}`);
    const blob = await glb.blob();
    // Save blob...
  } else if (data.status !== 'failed') {
    // Check again in 2 seconds
    setTimeout(checkStatus, 2000);
  }
};

checkStatus();
```

### Using the Python Client

```python
from client_example import TrellisClient

# Initialize client
client = TrellisClient("http://localhost:8000")

# Convert image
result_path = client.convert_image("image.jpg", seed=1)
print(f"Saved to: {result_path}")
```

## Configuration

### Choose Processing Method

Edit `main.py` in the `process_image()` function:

**Option 1: Proxy to HuggingFace (No GPU needed)**
```python
# Uncomment the HuggingFace block
from gradio_client import Client, handle_file
client = Client("JeffreyXiang/TRELLIS")
result = client.predict(...)
```

**Option 2: Local TRELLIS (Requires GPU)**
```python
# Uncomment the local TRELLIS block
from trellis.pipelines import TrellisImageTo3DPipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained(...)
```

## Deployment

### Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Fly.io

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly deploy
```

### Heroku

```bash
# Login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main
```

### AWS Lambda (Serverless)

For serverless, use Mangum adapter:

```python
from mangum import Mangum
handler = Mangum(app)
```

Then deploy with AWS SAM or Serverless Framework.

## Environment Variables

```bash
# Optional configuration
API_KEY=your-secret-key           # Enable API key authentication
MAX_FILE_SIZE=10485760            # Max upload size (10MB)
CLEANUP_AFTER_SECONDS=3600        # Delete files after 1 hour
HF_TOKEN=your-hf-token            # HuggingFace token (if needed)
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Security

For production:

1. **Enable authentication**:
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/api/convert")
async def convert_image(api_key: str = Security(api_key_header), ...):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ...
```

2. **Rate limiting**: Use `slowapi`
3. **CORS**: Configure allowed origins
4. **File validation**: Add size and type checks
5. **Use HTTPS** in production

## Performance

### Optimization Tips

1. **Use Redis** for job storage instead of in-memory dict
2. **Add Celery** for distributed task processing
3. **Use object storage** (S3, GCS) for files
4. **Enable caching** for common requests
5. **Add CDN** for serving GLB files

### Scaling

```yaml
# docker-compose.yml for scaling
services:
  trellis-api:
    build: .
    scale: 3  # Run 3 instances

  redis:
    image: redis:alpine

  nginx:
    image: nginx:alpine
    # Load balancer config
```

## Monitoring

Add health checks and metrics:

```python
from prometheus_client import Counter, Histogram

requests_total = Counter('requests_total', 'Total requests')
processing_time = Histogram('processing_seconds', 'Processing time')

@app.post("/api/convert")
async def convert_image(...):
    requests_total.inc()
    with processing_time.time():
        # ... processing
```

## Troubleshooting

### Port already in use
```bash
# Change port
uvicorn main:app --port 8001
```

### HuggingFace timeout
```python
# Increase timeout in gradio_client
client = Client("JeffreyXiang/TRELLIS", timeout=300)
```

### Out of memory
- Limit concurrent jobs
- Use queue system
- Scale horizontally

## License

MIT License - See main project LICENSE

## Resources

- **TRELLIS**: https://github.com/microsoft/TRELLIS
- **FastAPI**: https://fastapi.tiangolo.com
- **Gradio Client**: https://gradio.app/docs
