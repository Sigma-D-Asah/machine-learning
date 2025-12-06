"""
Test script untuk Machine Failure Prediction API
"""
import requests
import json

# Base URL
BASE_URL = "http://0.0.0.0:8000"

# Sample data untuk testing
sample_data = {
    "product_id": "L47340",
    "type": "L",
    "air_temperature": 298.4,
    "process_temperature": 308.2,
    "rotational_speed": 1282,
    "torque": 60.7,
    "tool_wear": 216
}

def print_response(title, response):
    """Helper function to print formatted response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"{'='*60}\n")


def test_health_check():
    """Test health check endpoint"""
    print("\nTesting Health Check...")
    response = requests.get(f"{BASE_URL}/api/v1/failure/health")
    print_response("HEALTH CHECK", response)
    return response.status_code == 200


def test_root():
    """Test root endpoint"""
    print("\nTesting Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print_response("ROOT ENDPOINT", response)
    return response.status_code == 200


def test_binary_prediction(data, title="Binary Prediction"):
    """Test binary prediction endpoint"""
    print(f"\nTesting Binary Prediction with {title}...")
    response = requests.post(
        f"{BASE_URL}/api/v1/failure/predict/binary",
        json=data
    )
    print_response(title, response)
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['data']['prediction']
        label = result['data']['prediction_label']
        print(f"Prediction: {prediction} ({label})")
    
    return response.status_code == 200


def test_multiclass_prediction(data, title="Multiclass Prediction"):
    """Test multiclass prediction endpoint"""
    print(f"\nTesting Multiclass Prediction with {title}...")
    response = requests.post(
        f"{BASE_URL}/api/v1/failure/predict/type",
        json=data
    )
    print_response(title, response)
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['data']['prediction']
        confidence = result['data']['confidence']
        ambiguous = result['data'].get('ambiguous', False)
        top_k = result['data'].get('top_k')
        suggested_override = result['data'].get('suggested_override')
        print(f"Predicted Failure Type: {prediction} (confidence: {confidence})")
        print(f"Ambiguous: {ambiguous}")
        if top_k:
            print(f"Top-K: {top_k}")
        if suggested_override:
            print(f"Suggested Override: {suggested_override}")
    
    return response.status_code == 200


def test_validation_error():
    """Test validation error with invalid data"""
    print("\nTesting Validation Error...")
    invalid_data = {
        "product_id": "M14860",
        "type": "M",
        # Missing required fields
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/failure/predict/binary",
        json=invalid_data
    )
    print_response("Validation Error Test", response)
    return response.status_code == 422


def run_all_tests():
    """Run all tests"""
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Root endpoint
    tests_total += 1
    if test_root():
        tests_passed += 1
    
    # Test 2: Health check
    tests_total += 1
    if test_health_check():
        tests_passed += 1
    
    # Test 3: Binary prediction with normal data
    tests_total += 1
    if test_binary_prediction(sample_data, "Normal Data"):
        tests_passed += 1
    
    # Test 4: Multiclass prediction with normal data
    tests_total += 1
    if test_multiclass_prediction(sample_data, "Normal Data"):
        tests_passed += 1
    
    # Test 5: Validation error
    tests_total += 1
    if test_validation_error():
        tests_passed += 1
    
    # Print summary
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print(f"Success Rate: {(tests_passed/tests_total)*100:.1f}%")


if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to the API server!")
        print("Please make sure the server is running:")
        print("uvicorn src.server:app --reload --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\nERROR: {str(e)}")