#!/bin/bash
#
# End-to-End CLI Tests for TRELLIS API
#
# Prerequisites:
#   1. Docker stack running: docker-compose up -d
#   2. API accessible at BASE_URL
#   3. Test image available (will create one if not)
#
# Usage:
#   ./e2e_cli_tests.sh
#   API_KEY="your-key" BASE_URL="http://localhost:8000" ./e2e_cli_tests.sh
#

set -e

# Configuration
API_KEY="${API_KEY:-dev-key-12345}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
TIMEOUT_SECONDS=120
POLL_INTERVAL=2

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0
SKIPPED=0

# Helper functions
pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++))
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++))
    if [ -n "$2" ]; then
        echo -e "  ${RED}Error: $2${NC}"
    fi
}

skip() {
    echo -e "${YELLOW}○ $1 (skipped)${NC}"
    ((SKIPPED++))
}

info() {
    echo -e "${BLUE}→ $1${NC}"
}

# Create test image if needed
create_test_image() {
    local img_path="/tmp/test_image.jpg"

    if [ -f "$img_path" ]; then
        return 0
    fi

    # Try Python/PIL first
    if command -v python3 &> /dev/null; then
        python3 -c "
from PIL import Image
img = Image.new('RGB', (100, 100), color='red')
img.save('$img_path', 'JPEG')
" 2>/dev/null && return 0
    fi

    # Try ImageMagick
    if command -v convert &> /dev/null; then
        convert -size 100x100 xc:red "$img_path" 2>/dev/null && return 0
    fi

    # Create minimal JPEG manually (hacky but works)
    echo -e '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09\x08\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9telecast,01444\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd5P\x00\x00\x00\xff\xd9' > "$img_path"

    return 0
}

# Wait for job completion
wait_for_job() {
    local job_id="$1"
    local max_attempts=$((TIMEOUT_SECONDS / POLL_INTERVAL))

    for ((i=1; i<=max_attempts; i++)); do
        response=$(curl -s -H "Authorization: Bearer $API_KEY" "$BASE_URL/api/v1/jobs/$job_id")
        status=$(echo "$response" | jq -r '.status' 2>/dev/null)

        case "$status" in
            "completed")
                echo "$response"
                return 0
                ;;
            "failed")
                echo "$response"
                return 1
                ;;
            "cancelled")
                echo "$response"
                return 2
                ;;
            *)
                progress=$(echo "$response" | jq -r '.progress // 0' 2>/dev/null)
                echo -ne "  Status: $status, Progress: ${progress}% (attempt $i/$max_attempts)\r"
                sleep $POLL_INTERVAL
                ;;
        esac
    done

    echo "Timeout waiting for job $job_id"
    return 3
}

# =============================================================================
# Tests
# =============================================================================

echo ""
echo "========================================="
echo "TRELLIS API End-to-End Tests"
echo "========================================="
echo "Base URL: $BASE_URL"
echo "API Key: ${API_KEY:0:8}..."
echo ""

# Create test image
info "Creating test image..."
create_test_image

# -----------------------------------------------------------------------------
# Test 1: Health Check
# -----------------------------------------------------------------------------
info "Test 1: Health endpoint"
HEALTH=$(curl -s "$BASE_URL/health" 2>&1)
if echo "$HEALTH" | grep -q '"status"'; then
    pass "Health check"
else
    fail "Health check" "$HEALTH"
fi

# -----------------------------------------------------------------------------
# Test 2: Root Endpoint
# -----------------------------------------------------------------------------
info "Test 2: Root endpoint"
ROOT=$(curl -s "$BASE_URL/" 2>&1)
if echo "$ROOT" | grep -q '"endpoints"'; then
    pass "Root endpoint"
else
    fail "Root endpoint" "$ROOT"
fi

# -----------------------------------------------------------------------------
# Test 3: Auth Required
# -----------------------------------------------------------------------------
info "Test 3: Authentication required"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/rembg" -X POST 2>&1)
if [ "$STATUS" == "401" ]; then
    pass "Auth required (401)"
else
    fail "Auth required" "Expected 401, got $STATUS"
fi

# -----------------------------------------------------------------------------
# Test 4: Invalid Auth
# -----------------------------------------------------------------------------
info "Test 4: Invalid authentication"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer invalid-key" \
    "$BASE_URL/api/v1/rembg" -X POST 2>&1)
if [ "$STATUS" == "401" ]; then
    pass "Invalid auth rejected (401)"
else
    fail "Invalid auth rejected" "Expected 401, got $STATUS"
fi

# -----------------------------------------------------------------------------
# Test 5: RemBG Endpoint
# -----------------------------------------------------------------------------
info "Test 5: RemBG endpoint"
REMBG_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -F "files=@/tmp/test_image.jpg" \
    "$BASE_URL/api/v1/rembg" 2>&1)

JOB_ID=$(echo "$REMBG_RESPONSE" | jq -r '.job_id' 2>/dev/null)
if [ -n "$JOB_ID" ] && [ "$JOB_ID" != "null" ]; then
    pass "RemBG job created: $JOB_ID"

    # Test 6: Job Status Polling
    info "Test 6: Job status polling"
    echo ""
    JOB_RESULT=$(wait_for_job "$JOB_ID")
    JOB_EXIT=$?
    echo ""

    case $JOB_EXIT in
        0)
            pass "Job completed"
            DOWNLOAD_URL=$(echo "$JOB_RESULT" | jq -r '.download_urls[0]' 2>/dev/null)

            # Test 7: Download Result
            if [ -n "$DOWNLOAD_URL" ] && [ "$DOWNLOAD_URL" != "null" ]; then
                info "Test 7: Download result"
                HTTP_CODE=$(curl -s -o /tmp/result.png -w "%{http_code}" \
                    -H "Authorization: Bearer $API_KEY" \
                    "$BASE_URL$DOWNLOAD_URL" 2>&1)
                if [ "$HTTP_CODE" == "200" ]; then
                    pass "Download successful"
                else
                    fail "Download failed" "HTTP $HTTP_CODE"
                fi
            else
                skip "Download (no URL)"
            fi
            ;;
        1)
            ERROR=$(echo "$JOB_RESULT" | jq -r '.error' 2>/dev/null)
            fail "Job failed" "$ERROR"
            skip "Download (job failed)"
            ;;
        2)
            fail "Job cancelled unexpectedly"
            skip "Download (job cancelled)"
            ;;
        3)
            fail "Job timeout"
            skip "Download (job timeout)"
            ;;
    esac
else
    fail "RemBG endpoint" "$REMBG_RESPONSE"
    skip "Job status polling"
    skip "Download result"
fi

# -----------------------------------------------------------------------------
# Test 8: TRELLIS Endpoint
# -----------------------------------------------------------------------------
info "Test 8: TRELLIS endpoint"
TRELLIS_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -F "files=@/tmp/test_image.jpg" \
    -F "seed=42" \
    "$BASE_URL/api/v1/trellis" 2>&1)

TRELLIS_JOB=$(echo "$TRELLIS_RESPONSE" | jq -r '.job_id' 2>/dev/null)
if [ -n "$TRELLIS_JOB" ] && [ "$TRELLIS_JOB" != "null" ]; then
    pass "TRELLIS job created: $TRELLIS_JOB"
else
    # TRELLIS might fail if HuggingFace space is down
    ERROR=$(echo "$TRELLIS_RESPONSE" | jq -r '.detail // .error // "Unknown error"' 2>/dev/null)
    skip "TRELLIS endpoint (may need HF connection): $ERROR"
fi

# -----------------------------------------------------------------------------
# Test 9: Job Deletion
# -----------------------------------------------------------------------------
info "Test 9: Job deletion"
# Create a job to delete
DELETE_JOB_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -F "files=@/tmp/test_image.jpg" \
    "$BASE_URL/api/v1/rembg" 2>&1)

DELETE_JOB_ID=$(echo "$DELETE_JOB_RESPONSE" | jq -r '.job_id' 2>/dev/null)
if [ -n "$DELETE_JOB_ID" ] && [ "$DELETE_JOB_ID" != "null" ]; then
    sleep 1  # Let the job start

    DELETE_RESPONSE=$(curl -s -X DELETE \
        -H "Authorization: Bearer $API_KEY" \
        "$BASE_URL/api/v1/jobs/$DELETE_JOB_ID" 2>&1)

    if echo "$DELETE_RESPONSE" | grep -q '"message"'; then
        pass "Job deleted"

        # Verify it's gone
        VERIFY=$(curl -s \
            -H "Authorization: Bearer $API_KEY" \
            "$BASE_URL/api/v1/jobs/$DELETE_JOB_ID" 2>&1)
        if echo "$VERIFY" | grep -q "not found"; then
            pass "Job deletion verified"
        else
            fail "Job deletion verification" "Job still exists"
        fi
    else
        fail "Job deletion" "$DELETE_RESPONSE"
        skip "Job deletion verification"
    fi
else
    fail "Job deletion" "Could not create test job"
    skip "Job deletion verification"
fi

# -----------------------------------------------------------------------------
# Test 10: Non-existent Job
# -----------------------------------------------------------------------------
info "Test 10: Non-existent job"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $API_KEY" \
    "$BASE_URL/api/v1/jobs/non-existent-job-id" 2>&1)
if [ "$STATUS" == "404" ]; then
    pass "Non-existent job returns 404"
else
    fail "Non-existent job" "Expected 404, got $STATUS"
fi

# -----------------------------------------------------------------------------
# Test 11: X-API-Key Header
# -----------------------------------------------------------------------------
info "Test 11: X-API-Key header"
HEALTH_XAPI=$(curl -s \
    -H "X-API-Key: $API_KEY" \
    "$BASE_URL/api/v1/jobs/test" 2>&1)
# Just checking that X-API-Key is accepted (will get 404 for the job)
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "X-API-Key: $API_KEY" \
    "$BASE_URL/api/v1/jobs/test" 2>&1)
if [ "$STATUS" == "404" ]; then
    pass "X-API-Key header works"
else
    fail "X-API-Key header" "Expected 404, got $STATUS"
fi

# -----------------------------------------------------------------------------
# Test 12: Invalid File Format
# -----------------------------------------------------------------------------
info "Test 12: Invalid file format"
echo "not an image" > /tmp/test.txt
INVALID_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -F "files=@/tmp/test.txt" \
    "$BASE_URL/api/v1/rembg" 2>&1)
rm -f /tmp/test.txt
if [ "$INVALID_RESPONSE" == "400" ]; then
    pass "Invalid format rejected (400)"
else
    fail "Invalid format rejection" "Expected 400, got $INVALID_RESPONSE"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================="
echo "E2E Test Summary"
echo "========================================="
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo ""

# Cleanup
rm -f /tmp/result.png

# Exit with error if any tests failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
