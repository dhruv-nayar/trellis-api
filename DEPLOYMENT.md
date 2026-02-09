# Deployment Guide - TRELLIS API

Step-by-step guides for deploying your API to the cloud.

---

## 🚀 Option 1: Railway (Easiest - Recommended)

**Free tier**: 500 hours/month, credit card required
**Best for**: Quick deployment, automatic HTTPS

### Step 1: Prepare Your Code

```bash
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool/api
```

Create `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health"
  }
}
```

### Step 2: Install Railway CLI

```bash
# Mac
brew install railway

# Or with npm
npm i -g @railway/cli

# Or with curl
curl -fsSL https://railway.app/install.sh | sh
```

### Step 3: Deploy

```bash
# Login
railway login

# Initialize project (run in api directory)
railway init

# Deploy
railway up

# Get URL
railway domain
```

Your API will be live at: `https://your-app.railway.app`

### Step 4: Configure Environment (Optional)

```bash
# Add environment variables if needed
railway variables set API_KEY=your-secret-key
```

**Done!** Your API is live 🎉

---

## 🪂 Option 2: Fly.io (Free Tier Available)

**Free tier**: 3 small VMs, 160GB bandwidth/month
**Best for**: Better free tier, multiple regions

### Step 1: Install Fly CLI

```bash
# Mac
brew install flyctl

# Or with curl
curl -L https://fly.io/install.sh | sh
```

### Step 2: Login

```bash
fly auth login
```

### Step 3: Create `fly.toml`

```toml
app = "trellis-api"  # Change this to your desired name

[build]
  builder = "paketobuildpacks/builder:base"

[env]
  PORT = "8000"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 512
```

### Step 4: Deploy

```bash
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool/api

# Launch (creates app)
fly launch --no-deploy

# Deploy
fly deploy

# Get URL
fly apps list
```

Your API will be live at: `https://trellis-api.fly.dev`

### Step 5: Check Status

```bash
fly status
fly logs
```

**Done!** 🎉

---

## 🎨 Option 3: Render (Very Simple)

**Free tier**: 750 hours/month, auto-sleep after inactivity
**Best for**: Simplest deployment (web UI)

### Step 1: Push to GitHub

```bash
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool

# Initialize git if not already
git init
git add .
git commit -m "Initial commit"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/trellis-api.git
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com)
2. Sign up / Login (free)
3. Click **"New +"** → **"Web Service"**
4. Connect your GitHub repo
5. Configure:
   - **Name**: trellis-api
   - **Root Directory**: `api`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Click **"Create Web Service"**

Your API will be live at: `https://trellis-api.onrender.com`

**Done!** 🎉

---

## 🐳 Option 4: Docker + Any Cloud

Deploy using Docker to AWS, GCP, Azure, DigitalOcean, etc.

### Step 1: Build Docker Image

```bash
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool/api

# Build
docker build -t trellis-api .

# Test locally
docker run -p 8000:8000 trellis-api
```

### Step 2: Push to Registry

**Docker Hub:**
```bash
# Login
docker login

# Tag
docker tag trellis-api YOUR_USERNAME/trellis-api:latest

# Push
docker push YOUR_USERNAME/trellis-api:latest
```

**GitHub Container Registry:**
```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag
docker tag trellis-api ghcr.io/USERNAME/trellis-api:latest

# Push
docker push ghcr.io/USERNAME/trellis-api:latest
```

### Step 3: Deploy to Cloud

**AWS ECS:**
```bash
# Install AWS CLI
brew install awscli

# Configure
aws configure

# Create task definition, service, etc.
# (See AWS ECS documentation)
```

**Google Cloud Run:**
```bash
# Install gcloud
brew install google-cloud-sdk

# Deploy
gcloud run deploy trellis-api \
  --image YOUR_USERNAME/trellis-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**DigitalOcean App Platform:**
1. Go to [DigitalOcean](https://www.digitalocean.com)
2. Create App → Docker Hub
3. Enter: `YOUR_USERNAME/trellis-api`
4. Deploy

---

## 🔧 Option 5: Heroku (Classic)

**Note**: Heroku removed free tier. Starts at $5/month.

### Step 1: Create `Procfile`

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Step 2: Deploy

```bash
# Install Heroku CLI
brew install heroku/brew/heroku

# Login
heroku login

# Create app
heroku create trellis-api

# Deploy
git push heroku main

# Open
heroku open
```

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

### 1. Configure Backend in `main.py`

Make sure HuggingFace block is uncommented:

```python
# Line ~150 in main.py
from gradio_client import Client, handle_file
client = Client("JeffreyXiang/TRELLIS")
result = client.predict(...)
```

### 2. Add `.gitignore`

Create `api/.gitignore`:
```
__pycache__/
*.pyc
.env
uploads/
outputs/
*.glb
.DS_Store
```

### 3. Test Locally First

```bash
python main.py
# Visit http://localhost:8000/docs
```

### 4. Environment Variables (Production)

Set these in your platform:

```bash
# Optional
API_KEY=your-secret-key
MAX_FILE_SIZE=10485760
CLEANUP_AFTER_SECONDS=3600
```

---

## 🔐 Production Security

### 1. Add API Key Authentication

Edit `main.py`:

```python
from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
import os

API_KEY = os.getenv("API_KEY", "change-me-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return api_key

@app.post("/api/convert")
async def convert_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    seed: int = 1,
    api_key: str = Depends(verify_api_key)  # Add this
):
    # ... rest of code
```

Then set `API_KEY` environment variable in your platform.

### 2. Add Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/convert")
@limiter.limit("10/minute")  # Max 10 requests per minute
async def convert_image(request: Request, ...):
    # ... rest of code
```

### 3. Configure CORS Properly

```python
# In main.py, update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ],  # Replace with your domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 🎯 After Deployment

### 1. Test Your API

```bash
# Replace with your deployed URL
API_URL="https://your-app.railway.app"

# Test health
curl $API_URL/health

# Test conversion
curl -X POST "$API_URL/api/convert" \
  -F "file=@test.jpg" \
  -F "seed=1"
```

### 2. Update Web UI

Edit `index.html` line 110:

```javascript
// Change this to your deployed URL
<input type="text" id="api-url" value="https://your-app.railway.app">
```

### 3. Monitor

Check logs:

**Railway:**
```bash
railway logs
```

**Fly.io:**
```bash
fly logs
```

**Render:**
Visit dashboard → Logs tab

---

## 📊 Comparison Table

| Platform | Free Tier | Setup | Speed | Best For |
|----------|-----------|-------|-------|----------|
| **Railway** | 500h/mo | ⭐⭐⭐⭐⭐ | Fast | Quick deploy |
| **Fly.io** | 3 VMs | ⭐⭐⭐⭐ | Fast | Best free tier |
| **Render** | 750h/mo | ⭐⭐⭐⭐⭐ | Slow start | Simple UI |
| **Heroku** | None | ⭐⭐⭐⭐ | Fast | Classic option |
| **Docker** | Varies | ⭐⭐⭐ | Varies | Flexibility |

---

## 🆘 Troubleshooting

### Build Fails

**Issue**: Dependencies won't install

**Fix**: Ensure `requirements.txt` is correct
```bash
pip freeze > requirements.txt
```

### App Crashes on Start

**Issue**: Port configuration

**Fix**: Use `$PORT` environment variable
```python
# In main.py
import os
port = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```

### Timeout Issues

**Issue**: Request timeouts on HuggingFace

**Fix**: Increase timeout in client
```python
client = Client("JeffreyXiang/TRELLIS", timeout=300)
```

### Memory Issues

**Issue**: Out of memory

**Fix**: Upgrade plan or optimize
- Delete uploaded files immediately
- Clear outputs after download
- Use smaller VM size initially

---

## 🎉 You're Live!

Your API is now deployed and accessible worldwide!

**Share your API:**
```
https://your-app.railway.app/docs
```

**Test with cURL:**
```bash
curl https://your-app.railway.app/health
```

**Integrate into apps:**
```javascript
fetch('https://your-app.railway.app/api/convert', {...})
```

Need help? Check the troubleshooting section or deploy logs!
