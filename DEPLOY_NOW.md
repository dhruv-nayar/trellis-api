# Deploy in 5 Minutes! 🚀

The absolute quickest way to get your API online.

---

## Railway (Recommended - Easiest)

### Step 1: Install Railway CLI

**Mac:**
```bash
brew install railway
```

**Linux/Mac (alternative):**
```bash
curl -fsSL https://railway.app/install.sh | sh
```

**Windows:**
```bash
npm i -g @railway/cli
```

### Step 2: Deploy

```bash
# Go to api directory
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool/api

# Login to Railway (opens browser)
railway login

# Initialize project
railway init

# Deploy!
railway up

# Get your URL
railway domain
```

**That's it!** Your API is live at `https://your-app.railway.app` 🎉

### Step 3: Test

```bash
# Get your URL
URL=$(railway domain)

# Test it
curl $URL/health
```

---

## Alternative: Fly.io (Best Free Tier)

### Step 1: Install Fly CLI

**Mac:**
```bash
brew install flyctl
```

**Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Deploy

```bash
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool/api

# Login
fly auth login

# Launch and deploy
fly launch --now
```

**Done!** Your API is live at `https://your-app.fly.dev` 🎉

---

## No Code? Use Web UI

### Render (100% Web-Based)

1. **Push to GitHub first:**
```bash
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool
git init
git add .
git commit -m "Deploy API"

# Create repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/trellis-api.git
git push -u origin main
```

2. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - Click **"New +"** → **"Web Service"**
   - Connect GitHub repo
   - **Root Directory**: `api`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Click **"Create"**

**Done!** Live at `https://your-app.onrender.com` 🎉

---

## Quick Test Your Deployed API

Replace `YOUR_URL` with your actual URL:

```bash
# Health check
curl https://YOUR_URL/health

# Convert an image
curl -X POST "https://YOUR_URL/api/convert" \
  -F "file=@test.jpg" \
  -F "seed=1"

# Check status (use job_id from above)
curl "https://YOUR_URL/api/status/JOB_ID"
```

---

## Update Your Web UI

Edit `index.html` line 110 with your deployed URL:

```html
<input type="text" id="api-url" value="https://your-app.railway.app">
```

---

## Need Help?

**Railway issues:**
```bash
railway logs  # Check logs
railway status  # Check status
```

**Fly.io issues:**
```bash
fly logs  # Check logs
fly status  # Check status
```

**Still stuck?**
- Check `DEPLOYMENT.md` for detailed troubleshooting
- Railway docs: https://docs.railway.app
- Fly.io docs: https://fly.io/docs

---

## What's Next?

✅ API is deployed
✅ Test it works
⬜ Add API key auth (see DEPLOYMENT.md)
⬜ Set up monitoring
⬜ Integrate into your app

**You're live! Start building!** 🚀
