import json
import numpy as np

def generate_spectrum():
    # Simulate 1024 points
    x = np.linspace(500, 1800, 1024)
    
    # Baseline
    baseline = 100 + 0.05 * x + 50 * np.exp(-(x - 500)/500)
    
    # Peaks (simulating a crystal)
    peaks = [
        (800, 500, 10),  # (position, height, width)
        (1085, 1200, 8), # Strong peak
        (1350, 300, 15),
        (1600, 400, 12)
    ]
    
    y = baseline
    for pos, height, width in peaks:
        y += height * np.exp(-(x - pos)**2 / (2 * width**2))
        
    # Noise
    y += np.random.normal(0, 5, 1024)
    
    # Normalize to 0-1 range for the model (optional, but good practice)
    # The model preprocessor handles normalization, but let's provide raw-ish counts
    y = np.clip(y, 0, None)
    
    return y.tolist()

if __name__ == "__main__":
    data = {
        "spectrum": generate_spectrum()
    }
    
    with open("example_request.json", "w") as f:
        json.dump(data, f, indent=2)
        
    print("Generated example_request.json with realistic Raman data.")
