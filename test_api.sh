#!/bin/bash

# TRELLIS API Test Script

API_URL="http://localhost:8000"
TEST_IMAGE="${1:-test.jpg}"

echo "==========================================="
echo "TRELLIS API Test Script"
echo "==========================================="
echo ""

# Check if image exists
if [ ! -f "$TEST_IMAGE" ]; then
    echo "❌ Test image not found: $TEST_IMAGE"
    echo ""
    echo "Usage: ./test_api.sh <image_path>"
    echo "Example: ./test_api.sh my-photo.jpg"
    exit 1
fi

echo "📤 Testing API with image: $TEST_IMAGE"
echo ""

# 1. Health check
echo "[1/4] Health check..."
curl -s "$API_URL/health" | python3 -m json.tool
echo ""

# 2. Upload image
echo "[2/4] Uploading image..."
RESPONSE=$(curl -s -X POST "$API_URL/api/convert" \
  -F "file=@$TEST_IMAGE" \
  -F "seed=1")

echo "$RESPONSE" | python3 -m json.tool
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

echo ""
echo "Job ID: $JOB_ID"
echo ""

# 3. Poll status
echo "[3/4] Waiting for completion..."
while true; do
    STATUS_RESPONSE=$(curl -s "$API_URL/api/status/$JOB_ID")
    STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
    MESSAGE=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('message', ''))")

    echo "   Status: $STATUS - $MESSAGE"

    if [ "$STATUS" = "completed" ]; then
        break
    elif [ "$STATUS" = "failed" ]; then
        echo ""
        echo "❌ Job failed!"
        echo "$STATUS_RESPONSE" | python3 -m json.tool
        exit 1
    fi

    sleep 2
done

echo ""

# 4. Download result
echo "[4/4] Downloading result..."
OUTPUT_FILE="output_${JOB_ID}.glb"
curl -s "$API_URL/api/download/$JOB_ID" -o "$OUTPUT_FILE"

if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo "==========================================="
    echo "✅ Success!"
    echo "==========================================="
    echo ""
    echo "Output: $OUTPUT_FILE"
    echo "Size: $FILE_SIZE"
    echo ""
    echo "View at: https://gltf-viewer.donmccurdy.com"
else
    echo ""
    echo "❌ Download failed"
    exit 1
fi
