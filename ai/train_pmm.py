#!/usr/bin/env python3
"""
train_pmm.py - Train initial PMM modes from historical Raman spectra.

This script initializes PMM mode prototypes (mu) from reference spectra
using k-means clustering, then saves as a baseline brain checkpoint.

Usage:
    python train_pmm.py --input spectra.csv --output checkpoints/baseline.npz --n-modes 8
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "bentoml_service"))
from pmm_brain import StaticPseudoModeMemory, RamanPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_spectra(path: Path) -> np.ndarray:
    """Load spectra from CSV or NPZ file."""
    if path.suffix == ".npz":
        data = np.load(str(path))
        # Assume spectra are stored under 'spectra' or 'data' key
        for key in ['spectra', 'data', 'X']:
            if key in data:
                return data[key]
        raise ValueError(f"No recognized key in {path}. Keys: {list(data.keys())}")
    elif path.suffix == ".csv":
        return np.loadtxt(str(path), delimiter=",", skiprows=1)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def preprocess_spectra(spectra: np.ndarray) -> torch.Tensor:
    """Preprocess all spectra using the Raman pipeline."""
    processed = []
    for i, spectrum in enumerate(spectra):
        try:
            x = RamanPreprocessor.preprocess(spectrum)
            processed.append(x)
        except Exception as e:
            logger.warning(f"Failed to preprocess spectrum {i}: {e}")
    
    if not processed:
        raise ValueError("No spectra could be preprocessed")
    
    return torch.stack(processed)


def train_initial_modes(
    spectra: torch.Tensor,
    n_modes: int = 8,
    latent_dim: int = 128,
    max_modes: int = 32
) -> StaticPseudoModeMemory:
    """
    Train initial PMM modes using k-means clustering.
    
    Args:
        spectra: Preprocessed spectra tensor (N, 1024)
        n_modes: Number of initial modes to create
        latent_dim: Latent dimension for encoder
        max_modes: Maximum modes for PMM
    
    Returns:
        Initialized PMM with trained modes
    """
    logger.info(f"Training {n_modes} initial modes from {len(spectra)} spectra")
    
    # Simple linear encoder (for training only)
    encoder = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, latent_dim)
    )
    
    # Encode all spectra
    with torch.no_grad():
        latents = encoder(spectra).numpy()
    
    # K-means clustering to find initial mode centers
    logger.info(f"Running k-means with k={n_modes}")
    kmeans = KMeans(n_clusters=n_modes, random_state=42, n_init=10)
    kmeans.fit(latents)
    
    # Create PMM and initialize modes
    pmm = StaticPseudoModeMemory(latent_dim=latent_dim, max_modes=max_modes, init_modes=n_modes)
    
    with torch.no_grad():
        # Set mu to cluster centers
        pmm.mu.data[:n_modes] = torch.from_numpy(kmeans.cluster_centers_).float()
        
        # Initialize occupancy based on cluster sizes
        _, counts = np.unique(kmeans.labels_, return_counts=True)
        occupancy = counts / counts.sum()
        pmm.occupancy[:n_modes] = torch.from_numpy(occupancy).float()
        
        # Reset active mask
        pmm.active_mask.zero_()
        pmm.active_mask[:n_modes] = True
    
    logger.info(f"Initialized {n_modes} modes with occupancies: {occupancy}")
    return pmm


def main():
    parser = argparse.ArgumentParser(description="Train PMM from reference spectra")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input spectra file (CSV or NPZ)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output checkpoint path")
    parser.add_argument("--n-modes", type=int, default=8, help="Number of initial modes")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--max-modes", type=int, default=32, help="Maximum modes")
    parser.add_argument("--org-id", type=str, default=None, help="Organization ID for versioning")
    args = parser.parse_args()
    
    # Load and preprocess spectra
    logger.info(f"Loading spectra from {args.input}")
    raw_spectra = load_spectra(args.input)
    logger.info(f"Loaded {len(raw_spectra)} spectra")
    
    spectra = preprocess_spectra(raw_spectra)
    logger.info(f"Preprocessed {len(spectra)} spectra")
    
    # Train initial modes
    pmm = train_initial_modes(
        spectra,
        n_modes=args.n_modes,
        latent_dim=args.latent_dim,
        max_modes=args.max_modes
    )
    
    # Save checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = pmm.save_state(args.output, org_id=args.org_id)
    logger.info(f"Saved baseline checkpoint to {checkpoint_path}")
    
    # Verify by loading
    pmm2 = StaticPseudoModeMemory(latent_dim=args.latent_dim, max_modes=args.max_modes)
    metadata = pmm2.load_state(checkpoint_path)
    logger.info(f"Verification: Loaded checkpoint with {pmm2.n_active} active modes")
    
    print(f"\n✓ Training complete. Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
