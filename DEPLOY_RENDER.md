# Deploy to Render (5 Minutes - No CLI!)

The easiest way to deploy your API - 100% web-based!

## 🚀 Quick Deploy

### Step 1: Push to GitHub

```bash
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool/api

# Run the GitHub script
./deploy-to-github.sh

# Then follow the instructions it prints
```

### Step 2: Deploy on Render

1. **Go to Render**: https://render.com
2. **Sign up/Login** (free account)
3. Click **"New +"** → **"Web Service"**
4. Click **"Connect account"** → Connect GitHub
5. **Select your repository**: `trellis-api`
6. **Configure**:
   - **Name**: trellis-api
   - **Region**: Oregon (closest to you)
   - **Branch**: main
   - **Root Directory**: Leave empty (or put `api` if you pushed the whole project)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run.py`
   - **Plan**: Free
7. Click **"Create Web Service"**

### Step 3: Wait (3-5 minutes)

Render will:
- Build your app
- Install dependencies
- Start your server
- Give you a URL: `https://trellis-api.onrender.com`

### Step 4: Test!

```bash
curl https://trellis-api.onrender.com/health
```

**Done!** 🎉

---

## 📋 If You Don't Have GitHub Yet

### Create GitHub Account
1. Go to https://github.com/signup
2. Create free account
3. Verify email

### Create Repository
1. Go to https://github.com/new
2. **Repository name**: trellis-api
3. **Public** or **Private** (your choice)
4. **Don't** check "Add README"
5. Click **"Create repository"**

### Push Your Code
```bash
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool/api

git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/trellis-api.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 🎯 Why Render is Better

✅ **No CLI** - Everything in web browser
✅ **Auto-deploy** - Push to GitHub = auto deploy
✅ **Free tier** - 750 hours/month
✅ **Logs in browser** - Easy to debug
✅ **SSL automatic** - HTTPS included
✅ **Reliable** - Used by many companies

---

## 📊 After Deployment

### View Logs
Just click **"Logs"** in the Render dashboard

### Update Your App
1. Make changes locally
2. Commit: `git commit -am "Update"`
3. Push: `git push`
4. Render auto-deploys!

### Your URLs
- **API**: https://trellis-api.onrender.com
- **Docs**: https://trellis-api.onrender.com/docs
- **Health**: https://trellis-api.onrender.com/health

---

## ⚡ Quick Commands Reference

```bash
# Initial setup
cd /Users/dhruvnayar/Desktop/Projects/trellis-local-tool/api
./deploy-to-github.sh

# Later updates
git add .
git commit -m "Update API"
git push
```

---

## 🆘 Troubleshooting

### Build fails
Check Render logs in dashboard. Usually:
- Missing dependency in requirements.txt
- Wrong Python version

### App crashes
- Check logs for errors
- Verify `run.py` is present
- Check PORT variable

### Can't connect GitHub
- Make sure repository is public OR
- Give Render access to private repos

---

**This is WAY easier than Railway!** No CLI issues, everything visual. 🎉
