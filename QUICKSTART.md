# TRELLIS API - Quick Start

Get your API running in 5 minutes!

## 🚀 Option 1: Quick Start (Easiest)

Uses HuggingFace backend - **no GPU needed!**

### Step 1: Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### Step 2: Configure Backend

Edit `main.py` line ~150 in the `process_image()` function:

**Uncomment the HuggingFace block:**
```python
# OPTION 1: Proxy to HuggingFace (No GPU needed)
from gradio_client import Client, handle_file

logger.info(f"Job {job_id}: Connecting to HuggingFace Space...")
client = Client("JeffreyXiang/TRELLIS")

logger.info(f"Job {job_id}: Processing image...")
result = client.predict(
    image=handle_file(str(input_path)),
    seed=seed,
    api_name="/image_to_3d"
)

# Get GLB path from result
if isinstance(result, dict) and 'glb' in result:
    glb_path = result['glb']
elif isinstance(result, str):
    glb_path = result
elif isinstance(result, tuple):
    glb_path = result[0]
else:
    raise ValueError(f"Unexpected result format: {type(result)}")

# Copy result to output path
shutil.copy(glb_path, output_path)
```

**Comment out the local TRELLIS block** (Option 2)

### Step 3: Start Server

```bash
python main.py
```

Or:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Test It!

**Option A: Web Interface**
Open `index.html` in your browser and upload an image!

**Option B: Command Line**
```bash
./test_api.sh your-image.jpg
```

**Option C: Python Client**
```python
python client_example.py
```

**Option D: cURL**
```bash
# Upload
curl -X POST "http://localhost:8000/api/convert" \
  -F "file=@image.jpg" \
  -F "seed=1"

# Get status (use job_id from above)
curl "http://localhost:8000/api/status/YOUR_JOB_ID"

# Download
curl "http://localhost:8000/api/download/YOUR_JOB_ID" -o output.glb
```

---

## 🐳 Option 2: Docker

Even easier!

```bash
# Build and run
docker-compose up --build

# API will be at http://localhost:8000
```

Then test with `index.html` or the test script!

---

## 🔧 Option 3: Local TRELLIS (Requires GPU)

If you have a Linux machine with NVIDIA GPU:

### Step 1: Install TRELLIS

```bash
# Run the setup script from main project
cd ..
./scripts/setup_trellis.sh
```

### Step 2: Configure API

Edit `main.py` line ~150:

**Comment out HuggingFace block, uncomment local TRELLIS:**

```python
# OPTION 2: Local TRELLIS (Requires GPU and TRELLIS installed)
import sys
sys.path.insert(0, str(Path.home() / ".cache/trellis/repo"))

from trellis.pipelines import TrellisImageTo3DPipeline
from PIL import Image
import torch

logger.info(f"Job {job_id}: Loading TRELLIS model...")
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Job {job_id}: Processing image...")
image = Image.open(input_path).convert("RGB")
outputs = pipeline.run(image, seed=seed)

logger.info(f"Job {job_id}: Exporting GLB...")
if hasattr(outputs, 'mesh'):
    outputs.mesh.export(str(output_path))
else:
    raise ValueError("Could not extract mesh from outputs")
```

### Step 3: Start Server

```bash
python main.py
```

---

## 📊 API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/api/convert` | POST | Upload & convert image |
| `/api/status/{job_id}` | GET | Check job status |
| `/api/download/{job_id}` | GET | Download GLB file |
| `/docs` | GET | Swagger UI |

---

## 🎯 Testing

### 1. Web UI
Open `index.html` in browser → Upload image → Convert!

### 2. Test Script
```bash
./test_api.sh path/to/image.jpg
```

### 3. Python Client
```python
from client_example import TrellisClient

client = TrellisClient("http://localhost:8000")
result = client.convert_image("image.jpg", seed=1)
print(f"Result: {result}")
```

### 4. JavaScript
```javascript
const formData = new FormData();
formData.append('file', imageFile);

const res = await fetch('http://localhost:8000/api/convert?seed=1', {
  method: 'POST',
  body: formData
});
const { job_id } = await res.json();

// Poll for completion...
```

---

## 🚢 Deployment

### Railway
```bash
railway login
railway init
railway up
```

### Fly.io
```bash
fly launch
fly deploy
```

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### Docker on Cloud
```bash
# Build
docker build -t trellis-api .

# Push to registry
docker tag trellis-api your-registry/trellis-api
docker push your-registry/trellis-api

# Deploy to cloud (AWS/GCP/Azure)
```

---

## 🔐 Production Checklist

- [ ] Add API key authentication
- [ ] Configure CORS properly
- [ ] Add rate limiting
- [ ] Use Redis for job storage
- [ ] Add file size limits
- [ ] Enable HTTPS
- [ ] Set up monitoring
- [ ] Add error tracking (Sentry)
- [ ] Use object storage (S3) for files
- [ ] Add request logging

---

## 💡 Tips

**Performance:**
- HuggingFace backend: ~30-60s per image, no GPU needed
- Local TRELLIS: ~15-30s per image, requires GPU

**Scaling:**
- Use Docker Compose to run multiple instances
- Add Redis for shared job storage
- Use Celery for distributed processing
- Add load balancer (nginx)

**Debugging:**
- Check logs: API outputs to console
- Test endpoints: Visit `/docs` for Swagger UI
- Health check: `curl http://localhost:8000/health`

---

## 🆘 Troubleshooting

**Port in use:**
```bash
uvicorn main:app --port 8001
```

**Can't reach API:**
- Check firewall
- Ensure server is running
- Try `0.0.0.0` instead of `localhost`

**HuggingFace timeout:**
- Increase timeout in gradio_client
- Check HF Space status
- Try again (might be busy)

**Local TRELLIS not working:**
- Ensure TRELLIS is installed: `./scripts/setup_trellis.sh`
- Check GPU: `nvidia-smi`
- Verify imports work

---

## 📚 Next Steps

1. ✅ Get API running locally
2. Test with sample images
3. Integrate into your app
4. Deploy to cloud
5. Add authentication & monitoring
6. Scale as needed!

---

## 🎉 You're Ready!

Your API is now running. Start converting images to 3D!

**Quick test:**
```bash
curl -X POST http://localhost:8000/api/convert \
  -F "file=@test.jpg" -F "seed=1"
```

Happy building! 🚀
