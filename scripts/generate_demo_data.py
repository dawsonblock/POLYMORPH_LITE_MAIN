import numpy as np
import pandas as pd
import os

def generate_crystallization_data(output_file="data/crystallization_demo.csv", steps=100):
    """
    Generates a synthetic Raman dataset simulating crystallization.
    
    Simulates:
    - Amorphous peak (broad) decreasing
    - Crystalline peaks (sharp) appearing and growing
    - Baseline drift
    - Noise
    """
    wavelengths = np.linspace(400, 1800, 1024)
    data = []
    
    # Peak centers (cm-1 or nm, treating as index for simplicity here, mapped to wavelengths)
    # Let's say we track 3 main features
    
    for i in range(steps):
        progress = i / steps
        
        # 1. Amorphous background (decreasing)
        # Broad Gaussian centered at 800
        amorphous = 500 * (1 - progress) * np.exp(-((wavelengths - 800) ** 2) / (2 * 150 ** 2))
        
        # 2. Crystalline peaks (growing)
        # Sharp peaks at 600, 1000, 1400
        cryst_1 = 800 * progress * np.exp(-((wavelengths - 600) ** 2) / (2 * 10 ** 2))
        cryst_2 = 1200 * progress * np.exp(-((wavelengths - 1000) ** 2) / (2 * 10 ** 2))
        cryst_3 = 600 * progress * np.exp(-((wavelengths - 1400) ** 2) / (2 * 10 ** 2))
        
        # 3. Baseline (random drift)
        baseline = 100 + 10 * np.sin(i * 0.1) + 0.05 * wavelengths
        
        # 4. Noise
        noise = np.random.normal(0, 5, len(wavelengths))
        
        spectrum = amorphous + cryst_1 + cryst_2 + cryst_3 + baseline + noise
        
        # Store as row: [step, w1, w2, ..., wN]
        # Actually, standard format usually has wavelengths as headers or first col.
        # Let's just store intensities. Wavelengths are constant.
        data.append(spectrum)
        
    # Save to CSV
    # First row: Wavelengths
    # Subsequent rows: Intensities for each time step
    df = pd.DataFrame(data, columns=wavelengths)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Generated {steps} frames of crystallization data to {output_file}")

if __name__ == "__main__":
    generate_crystallization_data()
