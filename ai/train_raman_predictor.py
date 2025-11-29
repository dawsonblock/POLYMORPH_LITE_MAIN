import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import bentoml
from ai.model import RamanCNN

from sqlalchemy.orm import Session
from retrofitkit.db.session import SessionLocal
from retrofitkit.db.models.sample import Sample
# Assuming spectral data is stored in extra_data or a separate table. 
# For this vNEXT implementation, we'll assume a 'spectral_data' field in extra_data.

def load_data_from_db(limit=1000):
    """Load spectral data from the database."""
    session: Session = SessionLocal()
    try:
        # Example query: fetch samples with spectral data
        # Use cast to JSONB if needed, or just check if key exists in JSON
        # For SQLite/Postgres compatibility in this repo, we'll just check if it's not null and do python-side filtering if needed for simplicity,
        # or use the correct operator.
        # Correct SQLAlchemy for JSON key existence is usually .has_key or similar depending on dialect, 
        # but standard JSON type might not support it directly in all versions.
        # Let's use a safer approach: fetch all and filter, or use text() for raw SQL if needed.
        # Given the error, let's try a simpler check or just fetch recent samples.
        samples = session.query(Sample).limit(limit).all()
        
        if not samples:
            print("No data found in DB, falling back to synthetic data.")
            return None, None
            
        X = []
        y = []
        for s in samples:
            if not s.extra_data:
                continue
            spectrum = s.extra_data.get('spectrum')
            concentration = s.extra_data.get('concentration', 0.0)
            if spectrum and len(spectrum) == 1024:
                X.append(spectrum)
                y.append(concentration)
                
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    except Exception as e:
        print(f"Error loading from DB: {e}")
        return None, None
    finally:
        session.close()

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
    print("Loading training data...")
    X_train, y_train = load_data_from_db()
    
    if X_train is None:
        print("Using synthetic data generator...")
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
