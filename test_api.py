#!/usr/bin/env python3
import requests
import json

API_URL = "http://localhost:8000"

def test_prediction(year, month, description=""):
    url = f"{API_URL}/predict"
    payload = {"year": year, "month": month}

    print(f"\n{description}")
    print(f"Payload: {json.dumps(payload)} (types: year={type(year).__name__}, month={type(month).__name__})")

    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✓ Success: {response.json()}")
        else:
            print(f"✗ Error: {response.json()}")
    except Exception as e:
        print(f"✗ Exception: {e}")

def test_raw_json(year, month, description=""):
    url = f"{API_URL}/predict"
    json_str = json.dumps({"year": year, "month": month})

    print(f"\n{description}")
    print(f"Raw JSON string: {json_str}")

    try:
        response = requests.post(url, data=json_str, headers={"Content-Type": "application/json"})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✓ Success: {response.json()}")
        else:
            print(f"✗ Error: {response.json()}")
    except Exception as e:
        print(f"✗ Exception: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("API Test Script - Simulating Real HTTP Requests")
    print("=" * 70)

    print("\n[1] Homepage")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("[2] Valid Requests - Testing Different Month Formats")
    print("=" * 70)

    test_prediction(2021, 1, "Test 1: Integer month (1)")
    test_prediction(2021, 6, "Test 2: Integer month (6)")
    test_prediction(2023, 12, "Test 3: Integer month (12)")

    print("\n" + "=" * 70)
    print("[3] String Month Format - Now Supported!")
    print("=" * 70)

    test_prediction(2021, "01", "Test 4: String month ('01') - converts to int")
    test_prediction(2021, "1", "Test 5: String month ('1') - converts to int")
    test_prediction("2023", "06", "Test 6: Both as strings")

    print("\n" + "=" * 70)
    print("[4] Raw JSON Testing (Simulating Frontend/Postman)")
    print("=" * 70)

    test_raw_json(2021, 1, "Test 7: Raw JSON with int month")
    test_raw_json(2024, 6, "Test 8: Raw JSON with int month")

    print("\n" + "=" * 70)
    print("[5] Error Cases")
    print("=" * 70)

    test_prediction(2021, 13, "Test 9: Invalid month (13)")
    test_prediction(2021, 0, "Test 10: Invalid month (0)")
    test_prediction(2019, 1, "Test 11: Past date (2019)")
    test_prediction(2021, "abc", "Test 12: Invalid string format")

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)
    print("\nNOTE: API now accepts both integers and strings for year/month.")
    print("Examples: month=1, month='01', month='1' all work!")
    print("JSON format {'month': 01} is also valid (parsed as integer 1).")
