#!/usr/bin/env python3
"""
Test script for the prediction API
"""
import requests
import json

# Change this to your deployed URL when testing production
API_URL = "http://localhost:8000"

def test_prediction(year, month):
    """Test a single prediction - accepts both int and string for month"""
    url = f"{API_URL}/predict"
    payload = {"year": year, "month": month}

    # Display month properly whether it's int or string
    month_display = f"{int(month):02d}" if month else "??"
    print(f"\nTesting: {year}/{month_display}")
    print(f"Request: {json.dumps(payload)}")

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("API Test Script")
    print("=" * 60)

    # Test homepage
    print("\n1. Testing homepage:")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Error: {e}")

    # Test various predictions
    print("\n2. Testing predictions:")

    # Test 2021 data with integer
    test_prediction(2021, 1)

    # Test with string format "01" to verify API handles it
    test_prediction(2021, "01")

    # Test 2023 data
    test_prediction(2023, 6)

    # Test 2025 data
    test_prediction(2025, 12)

    # Test error handling
    print("\n3. Testing error handling:")

    # Invalid month
    test_prediction(2021, 13)

    # Past date (assuming training data ends at 2020)
    test_prediction(2019, 1)

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)
