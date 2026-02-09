#!/bin/bash

echo "📦 Preparing for GitHub..."
echo ""

cd "$(dirname "$0")"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit - TRELLIS API"
    echo ""
    echo "✅ Git repository initialized"
    echo ""
    echo "Next steps:"
    echo "1. Create a new repository on GitHub: https://github.com/new"
    echo "2. Name it: trellis-api"
    echo "3. Don't initialize with README"
    echo "4. Run these commands:"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/trellis-api.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
else
    echo "Git repository already exists"
    echo "Adding all files..."
    git add .
    git commit -m "Update API with fixes"
    echo ""
    echo "✅ Changes committed"
    echo ""
    echo "Push to GitHub:"
    echo "   git push"
fi
