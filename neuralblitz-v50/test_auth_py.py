#!/usr/bin/env python3
"""
NeuralBlitz v50 JWT Authentication Test
Tests the JWT authentication system with the Flask test client
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from applications.unified_api import app


def test_authentication():
    """Test JWT authentication system"""
    print("==========================================")
    print("NEURALBLITZ v50 JWT AUTHENTICATION TEST")
    print("==========================================\n")

    passed = 0
    failed = 0

    with app.test_client() as client:
        # Get tokens
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123", "grant_type": "password"},
        )
        admin_token = response.json.get("access_token", "")

        response = client.post(
            "/api/v1/auth/token",
            data={"username": "viewer", "password": "viewer123", "grant_type": "password"},
        )
        viewer_token = response.json.get("access_token", "")

        # Test 1: Public health endpoint
        print("1. Testing public health endpoint...")
        response = client.get("/api/v1/health")
        if response.status_code == 200 and response.json.get("status") == "healthy":
            print("   ✅ PASS: Health endpoint works")
            passed += 1
        else:
            print(f"   ❌ FAIL: Health endpoint failed (status: {response.status_code})")
            failed += 1

        # Test 2: Admin token generation
        print("\n2. Testing admin token generation...")
        if admin_token:
            print("   ✅ PASS: Admin token generated")
            passed += 1
        else:
            print(f"   ❌ FAIL: Admin token generation failed: {response.json}")
            failed += 1

        # Test 3: Viewer token generation
        print("\n3. Testing viewer token generation...")
        if viewer_token:
            print("   ✅ PASS: Viewer token generated")
            passed += 1
        else:
            print(f"   ❌ FAIL: Viewer token generation failed")
            failed += 1

        # Test 4: Protected endpoint with admin token
        print("\n4. Testing protected endpoint with admin token...")
        if admin_token:
            response = client.get(
                "/api/v1/metrics", headers={"Authorization": f"Bearer {admin_token}"}
            )
            if response.status_code == 200:
                print("   ✅ PASS: Admin can access protected endpoint")
                passed += 1
            else:
                print(
                    f"   ❌ FAIL: Admin cannot access protected endpoint (status: {response.status_code})"
                )
                failed += 1
        else:
            print("   ⏭ SKIP: No admin token available")
            failed += 1

        # Test 5: Unauthenticated request rejected
        print("\n5. Testing unauthenticated request rejection...")
        response = client.get("/api/v1/metrics")
        if response.status_code == 401:
            print("   ✅ PASS: Unauthenticated request rejected")
            passed += 1
        else:
            print(
                f"   ❌ FAIL: Unauthenticated request not rejected (status: {response.status_code})"
            )
            failed += 1

        # Test 6: Invalid token rejected
        print("\n6. Testing invalid token rejection...")
        response = client.get(
            "/api/v1/metrics", headers={"Authorization": "Bearer invalid_token_here"}
        )
        if response.status_code == 401:
            print("   ✅ PASS: Invalid token rejected")
            passed += 1
        else:
            print(f"   ❌ FAIL: Invalid token not rejected (status: {response.status_code})")
            failed += 1

        # Test 7: Demo credentials endpoint
        print("\n7. Testing demo credentials endpoint...")
        response = client.get("/api/v1/auth/demo")
        if response.status_code == 200 and "users" in response.json:
            print("   ✅ PASS: Demo credentials endpoint works")
            passed += 1
        else:
            print(f"   ❌ FAIL: Demo credentials endpoint failed (status: {response.status_code})")
            failed += 1

        # Test 8: Token introspection
        print("\n8. Testing token introspection...")
        if admin_token:
            response = client.post(
                "/api/v1/auth/introspect", headers={"Authorization": f"Bearer {admin_token}"}
            )
            if response.status_code == 200 and "active" in response.json:
                print("   ✅ PASS: Token introspection works")
                passed += 1
            else:
                print(f"   ❌ FAIL: Token introspection failed: {response.json}")
                failed += 1
        else:
            print("   ⏭ SKIP: No admin token available")
            failed += 1

        # Test 9: System status endpoint
        print("\n9. Testing system status endpoint...")
        if admin_token:
            response = client.get(
                "/api/v1/status", headers={"Authorization": f"Bearer {admin_token}"}
            )
            if response.status_code == 200:
                print("   ✅ PASS: System status accessible")
                passed += 1
            else:
                print(f"   ❌ FAIL: System status not accessible (status: {response.status_code})")
                failed += 1
        else:
            print("   ⏭ SKIP: No admin token available")
            failed += 1

        # Test 10: Quantum state endpoint
        print("\n10. Testing quantum state endpoint...")
        if admin_token:
            response = client.get(
                "/api/v1/quantum/state", headers={"Authorization": f"Bearer {admin_token}"}
            )
            if response.status_code == 200:
                print("   ✅ PASS: Quantum state accessible")
                passed += 1
            else:
                print(f"   ❌ FAIL: Quantum state not accessible (status: {response.status_code})")
                failed += 1
        else:
            print("   ⏭ SKIP: No admin token available")
            failed += 1

        # Test 11: Viewer scope enforcement
        print("\n11. Testing viewer scope enforcement...")
        if viewer_token:
            response = client.get(
                "/api/v1/metrics", headers={"Authorization": f"Bearer {viewer_token}"}
            )
            if response.status_code == 403:
                print("   ✅ PASS: Viewer cannot access metrics without scope")
                passed += 1
            else:
                print(f"   ❌ FAIL: Viewer scope not enforced (status: {response.status_code})")
                failed += 1
        else:
            print("   ⏭ SKIP: No viewer token available")
            failed += 1

        # Test 12: Invalid credentials rejected
        print("\n12. Testing invalid credentials rejection...")
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "invalid", "password": "wrong", "grant_type": "password"},
        )
        if response.status_code == 401:
            print("   ✅ PASS: Invalid credentials rejected")
            passed += 1
        else:
            print(f"   ❌ FAIL: Invalid credentials not rejected (status: {response.status_code})")
            failed += 1

        # Test 13: Viewer execute denied
        print("\n13. Testing viewer execute denial...")
        if viewer_token:
            response = client.post(
                "/api/v1/quantum/step", headers={"Authorization": f"Bearer {viewer_token}"}
            )
            if response.status_code == 403:
                print("   ✅ PASS: Viewer cannot execute without scope")
                passed += 1
            else:
                print(f"   ❌ FAIL: Viewer execute not denied (status: {response.status_code})")
                failed += 1
        else:
            print("   ⏭ SKIP: No viewer token available")
            failed += 1

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    print("=" * 50)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(test_authentication())
