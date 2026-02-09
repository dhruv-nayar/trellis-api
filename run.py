#!/usr/bin/env python3
"""
Startup script for Railway deployment
Handles PORT environment variable properly
"""
import os
import sys

def main():
    # Get PORT from environment, default to 8000
    port = int(os.environ.get("PORT", 8000))

    print(f"Starting server on port {port}...")

    # Import uvicorn and run
    import uvicorn
    from main import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
