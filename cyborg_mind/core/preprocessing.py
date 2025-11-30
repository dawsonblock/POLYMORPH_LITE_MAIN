import torch
import numpy as np
from scipy.signal import savgol_filter

class RamanPreprocessor:
    """Production Raman preprocessing pipeline — identical across training/inference"""
    
    @staticmethod
    def preprocess(spectrum: np.ndarray) -> torch.Tensor:
        x = spectrum.astype(np.float32)
        
        # Validate input length
        if len(x) < 10:
            # Return a zero tensor or handle gracefully
            return torch.zeros(1024, dtype=torch.float32)

        # 1. Baseline correction (asymmetric least squares)
        x = RamanPreprocessor._als_baseline(x)
        
        # 2. Cosmic ray removal (median filter)
        x = np.medfilt(x, kernel_size=5)
        
        # 3. Savitzky-Golay smoothing
        if len(x) >= 15:
            x = savgol_filter(x, window_length=15, polyorder=3, mode='nearest')
        
        # 4. Normalization (area under curve)
        area = np.trapz(x, dx=1.0)
        if area > 1e-6:  # Avoid division by zero
            x = x / area
        
        # 5. Range selection (500–1800 cm⁻¹ typical for organics)
        # Adjust indices based on your spectrometer calibration
        if len(x) > 900:
             x = x[100:900]  # example: 500–1800 cm⁻¹
        
        # 6. Pad/truncate to fixed size
        target_len = 1024
        if len(x) < target_len:
            x = np.pad(x, (0, target_len - len(x)), mode='constant')
        else:
            x = x[:target_len]
            
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def _als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction (Eilers & Boelens, 2005).
        """
        L = y.shape[0]
        if L < 3:
            return np.zeros_like(y)

        # Second-difference operator
        D = np.diff(np.eye(L), 2, axis=0)  # (L-2, L)
        w = np.ones(L)

        for _ in range(niter):
            W = np.diag(w)
            # Proper ALS system: Z is (L, L)
            Z = W + lam * (D.T @ D)
            # Solve Z z = W y
            z = np.linalg.solve(Z, w * y)

            # Update weights: penalize positive residuals more
            w = p * (y > z) + (1.0 - p) * (y <= z)

        return z
