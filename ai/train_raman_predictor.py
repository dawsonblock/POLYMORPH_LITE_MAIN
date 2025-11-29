import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import bentoml
from ai.model import RamanCNN

def generate_synthetic_data(num_samples=1000, input_size=1024):
    """Generate synthetic Raman spectra with varying peak heights."""
    X = []
    y = []
    wavelengths = np.linspace(400, 900, input_size)
    
    for _ in range(num_samples):
        concentration = np.random.rand()
        # Peak at 532nm, height proportional to concentration
        intensity = (concentration * 100) * np.exp(-((wavelengths - 532) ** 2) / (2 * 10 ** 2))
        # Add noise
        intensity += np.random.normal(0, 5, input_size)
        # Normalize
        intensity = intensity / 150.0 
        
        X.append(intensity)
        y.append(concentration)
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

def train():
    print("Generating synthetic data...")
    X_train, y_train = generate_synthetic_data()
    
    model = RamanCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model...")
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    print("Training complete.")
    
    # Save with BentoML
    bento_model = bentoml.pytorch.save_model("raman_predictor", model)
    print(f"Model saved: {bento_model.tag}")

if __name__ == "__main__":
    train()
