#!/usr/bin/env python3
import requests

API_URL = "http://localhost:8000"

def test_request(json_string, label):
    """Send raw JSON string exactly like curl does"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            data=json_string,
            headers={"Content-Type": "application/json"}
        )
        status = "✓" if response.status_code == 200 else "✗"
        result = response.json()
        print(f"{status} {label:<40} {json_string:<30} -> {result}")
    except Exception as e:
        print(f"✗ {label:<40} {json_string:<30} -> Exception: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("API Test - Raw JSON (simulates curl/Postman exactly)")
    print("=" * 100)

    # Homepage
    try:
        response = requests.get(f"{API_URL}/")
        print(f"\n✓ GET / -> {response.json()}")
    except Exception as e:
        print(f"\n✗ GET / -> {e}")

    print("\n" + "-" * 100)
    print("Valid Requests")
    print("-" * 100)
    test_request('{"year": 2021, "month": 1}', 'Integer (no leading zero)')
    test_request('{"year": 2021, "month": "01"}', 'String (with leading zero)')
    test_request('{"year": 2021, "month": "1"}', 'String (no leading zero)')
    test_request('{"year": "2023", "month": "06"}', 'Both as strings')

    print("\n" + "-" * 100)
    print("Invalid JSON (leading zeros in numbers)")
    print("-" * 100)
    test_request('{"year": 2021, "month": 01}', 'INVALID: leading zero 01')
    test_request('{"year": 2021, "month": 06}', 'INVALID: leading zero 06')

    print("\n" + "-" * 100)
    print("Error Cases (valid JSON, invalid values)")
    print("-" * 100)
    test_request('{"year": 2021, "month": 13}', 'Invalid month 13')
    test_request('{"year": 2021, "month": 0}', 'Invalid month 0')
    test_request('{"year": 2019, "month": 1}', 'Past date')

    print("\n" + "=" * 100)
    print("Summary: Use \"month\": 1 (integer) or \"month\": \"01\" (string), NOT \"month\": 01")
    print("=" * 100 + "\n")
