import requests
import numpy as np
import json

def test_infer():
    url = "http://localhost:3000/infer"
    headers = {"Content-Type": "application/json"}
    
    # Create a dummy spectrum
    spectrum = np.random.rand(1024).tolist()
    
    data = {"spectrum": spectrum}
    
    try:
        response = requests.post(url, json=data, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("SUCCESS: Service is working.")
        else:
            print("FAILURE: Service returned error.")
            
    except Exception as e:
        print(f"ERROR: Failed to connect to service: {e}")

if __name__ == "__main__":
    test_infer()
