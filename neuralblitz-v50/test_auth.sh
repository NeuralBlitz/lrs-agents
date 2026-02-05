#!/bin/bash
# NeuralBlitz v50 JWT Authentication Test Suite
# Tests authentication endpoints with various scenarios

API_URL="${API_URL:-http://localhost:5000}"
SSL_API_URL="${SSL_API_URL:-https://localhost:5443}"

echo "=========================================="
echo "NEURALBLITZ v50 JWT AUTHENTICATION TEST"
echo "=========================================="
echo ""
echo "API URL: $API_URL"
echo "SSL API URL: $SSL_API_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

# Test function
run_test() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    echo -n "Testing: $name... "
    result=$(eval "$command" 2>/dev/null)
    
    if [[ "$result" == *"$expected"* ]]; then
        echo -e "${GREEN}PASS${NC}"
        ((passed++))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Expected: $expected"
        echo "  Got: $result"
        ((failed++))
    fi
}

# Demo credentials
echo "1. Getting demo credentials..."
demo_response=$(curl -s "$API_URL/api/v1/auth/demo")
echo "$demo_response" | head -c 100
echo "..."
echo ""

# Get tokens for different users
echo "2. Testing authentication..."
admin_token=$(curl -s -X POST "$API_URL/api/v1/auth/token" \
    -d "username=admin" \
    -d "password=admin123" \
    -d "grant_type=password" | jq -r '.access_token')

viewer_token=$(curl -s -X POST "$API_URL/api/v1/auth/token" \
    -d "username=viewer" \
    -d "password=viewer123" \
    -d "grant_type=password" | jq -r '.access_token')

operator_token=$(curl -s -X POST "$API_URL/api/v1/auth/token" \
    -d "username=operator" \
    -d "password=operator123" \
    -d "grant_type=password" | jq -r '.access_token')

# Test token generation
echo ""
echo "3. Testing token generation..."
if [[ "$admin_token" != "null" && ${#admin_token} -gt 50 ]]; then
    echo -e "${GREEN}PASS${NC}: Admin token generated successfully"
    ((passed++))
else
    echo -e "${RED}FAIL${NC}: Admin token generation failed"
    ((failed++))
fi

if [[ "$viewer_token" != "null" && ${#viewer_token} -gt 50 ]]; then
    echo -e "${GREEN}PASS${NC}: Viewer token generated successfully"
    ((passed++))
else
    echo -e "${RED}FAIL${NC}: Viewer token generation failed"
    ((failed++))
fi

# Test authenticated endpoints
echo ""
echo "4. Testing authenticated endpoints..."

# Test with admin token (has all scopes)
run_test "Admin can access metrics" \
    "curl -s -H 'Authorization: Bearer $admin_token' $API_URL/api/v1/metrics | jq -r '.quantum_coherence'" \
    "0."

# Test with viewer token (limited scopes)
run_test "Viewer can access metrics" \
    "curl -s -H 'Authorization: Bearer $viewer_token' $API_URL/api/v1/metrics | jq -r '.quantum_coherence'" \
    "0."

# Test without token (should fail)
run_test "Unauthenticated request rejected" \
    "curl -s $API_URL/api/v1/metrics | jq -r '.error'" \
    "missing_authorization_header"

# Test with invalid token
run_test "Invalid token rejected" \
    "curl -s -H 'Authorization: Bearer invalid_token_here' $API_URL/api/v1/metrics | jq -r '.error'" \
    "invalid_token"

# Test introspection
echo ""
echo "5. Testing token introspection..."
introspect_response=$(curl -s -X POST "$API_URL/api/v1/auth/introspect" \
    -H "Authorization: Bearer $admin_token")
    
if [[ "$introspect_response" == *"active\": true"* ]]; then
    echo -e "${GREEN}PASS${NC}: Token introspection works"
    ((passed++))
else
    echo -e "${RED}FAIL${NC}: Token introspection failed"
    echo "  Response: $introspect_response"
    ((failed++))
fi

# Test scope enforcement
echo ""
echo "6. Testing scope enforcement..."

# Viewer should not be able to execute (no execute scope)
execute_response=$(curl -s -X POST -H "Authorization: Bearer $viewer_token" \
    -H "Content-Type: application/json" \
    -d '{"steps": 1}' \
    "$API_URL/api/v1/quantum/step" | jq -r '.error')

if [[ "$execute_response" == *"insufficient_scope"* ]]; then
    echo -e "${GREEN}PASS${NC}: Scope enforcement works"
    ((passed++))
else
    echo -e "${RED}FAIL${NC}: Scope enforcement failed"
    ((failed++))
fi

# Test invalid credentials
echo ""
echo "7. Testing invalid credential handling..."
wrong_password=$(curl -s -X POST "$API_URL/api/v1/auth/token" \
    -d "username=admin" \
    -d "password=wrongpassword" \
    -d "grant_type=password" | jq -r '.error')

if [[ "$wrong_password" == *"invalid_grant"* ]]; then
    echo -e "${GREEN}PASS${NC}: Invalid credentials rejected"
    ((passed++))
else
    echo -e "${RED}FAIL${NC}: Invalid credentials not rejected"
    ((failed++))
fi

# Test SSL endpoint (if available)
echo ""
echo "8. Testing SSL endpoint..."
ssl_available=false
if curl -sf --max-time 5 "$SSL_API_URL/api/v1/health" > /dev/null 2>&1; then
    ssl_available=true
    echo -e "${GREEN}INFO${NC}: SSL endpoint available"
    
    # Test SSL with token
    ssl_token=$(curl -s -X POST "$SSL_API_URL/api/v1/auth/token" \
        -d "username=admin" \
        -d "password=admin123" \
        -d "grant_type=password" | jq -r '.access_token')
    
    if [[ "$ssl_token" != "null" ]]; then
        echo -e "${GREEN}PASS${NC}: SSL authentication works"
        ((passed++))
    else
        echo -e "${RED}FAIL${NC}: SSL authentication failed"
        ((failed++))
    fi
else
    echo -e "${YELLOW}SKIP${NC}: SSL endpoint not available (expected in development mode)"
fi

# Summary
echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo -e "Passed: ${GREEN}$passed${NC}"
echo -e "Failed: ${RED}$failed${NC}"
echo ""

if [[ $failed -eq 0 ]]; then
    echo -e "${GREEN}ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}SOME TESTS FAILED${NC}"
    exit 1
fi
