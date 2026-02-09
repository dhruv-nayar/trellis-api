#!/bin/bash

echo "🚀 Deploying to Railway..."
echo ""

# Check if railway is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not installed"
    exit 1
fi

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "❌ Not logged in to Railway"
    echo "Run: railway login"
    exit 1
fi

echo "✅ Logged in to Railway"
echo ""

# Navigate to api directory
cd "$(dirname "$0")"

# Check if .railway directory exists
if [ ! -d ".railway" ]; then
    echo "⚠️  Not linked to Railway project"
    echo ""
    echo "Run this command first:"
    echo "  railway link"
    echo ""
    echo "Then select:"
    echo "  1. Workspace: Dhruv Nayar's Projects"
    echo "  2. Project: trellis-api"
    echo "  3. Environment: production"
    echo ""
    echo "After linking, run this script again."
    exit 1
fi

echo "📦 Uploading code to Railway..."
railway up --detach

echo ""
echo "✅ Deployment started!"
echo ""
echo "Check status:"
echo "  railway logs"
echo ""
echo "Your API will be at:"
echo "  https://trellis-api-production.up.railway.app"
