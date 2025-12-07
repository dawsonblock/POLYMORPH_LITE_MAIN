#!/usr/bin/env python3
"""
calibrate_raman.py - Per-instrument Raman calibration.

Computes offset and scale corrections for a specific spectrometer
based on a reference spectrum (e.g., polystyrene standard).

Usage:
    python calibrate_raman.py --reference polystyrene.csv --instrument SN12345 --output calibration/
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from scipy.signal import find_peaks, correlate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Known polystyrene reference peaks (cm^-1)
POLYSTYRENE_PEAKS = [621, 795, 1001, 1031, 1155, 1450, 1583, 1602]


def load_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from file.
    
    Returns:
        wavenumbers: Array of wavenumber values (cm^-1)
        intensities: Array of intensity values
    """
    data = np.loadtxt(str(path), delimiter=",", skiprows=1)
    if data.ndim == 1:
        # Assume equal spacing from 200-2000 cm^-1
        wavenumbers = np.linspace(200, 2000, len(data))
        intensities = data
    elif data.shape[1] == 2:
        wavenumbers = data[:, 0]
        intensities = data[:, 1]
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    
    return wavenumbers, intensities


def find_reference_peaks(wavenumbers: np.ndarray, intensities: np.ndarray) -> Dict[int, float]:
    """
    Find peaks in spectrum and match to known reference peaks.
    
    Returns:
        Dict mapping expected peak position to measured position
    """
    # Find peaks in measured spectrum
    peaks, properties = find_peaks(intensities, height=np.percentile(intensities, 80), distance=10)
    peak_positions = wavenumbers[peaks]
    
    # Match to known peaks
    matched = {}
    for expected in POLYSTYRENE_PEAKS:
        closest_idx = np.argmin(np.abs(peak_positions - expected))
        if abs(peak_positions[closest_idx] - expected) < 50:  # Within 50 cm^-1
            matched[expected] = float(peak_positions[closest_idx])
    
    return matched


def compute_calibration(matched_peaks: Dict[int, float]) -> Dict[str, float]:
    """
    Compute linear calibration from matched peaks.
    
    Returns:
        Dict with 'offset' and 'scale' corrections
    """
    if len(matched_peaks) < 2:
        raise ValueError(f"Need at least 2 matched peaks, found {len(matched_peaks)}")
    
    expected = np.array(list(matched_peaks.keys()))
    measured = np.array(list(matched_peaks.values()))
    
    # Linear fit: measured = scale * expected + offset
    # Rearranged: expected = (measured - offset) / scale
    coeffs = np.polyfit(measured, expected, 1)
    scale = coeffs[0]
    offset = coeffs[1]
    
    # Compute residuals
    corrected = measured * scale + offset
    residuals = expected - corrected
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    logger.info(f"Calibration: scale={scale:.6f}, offset={offset:.4f} cm^-1, RMSE={rmse:.4f} cm^-1")
    
    return {
        "scale": float(scale),
        "offset": float(offset),
        "rmse": float(rmse),
        "n_peaks": len(matched_peaks)
    }


def save_calibration(
    calibration: Dict[str, float],
    instrument_id: str,
    output_dir: Path
) -> Path:
    """Save calibration profile to JSON file."""
    profile = {
        "instrument_id": instrument_id,
        "calibration": calibration,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_peaks": POLYSTYRENE_PEAKS,
        "version": "1.0"
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{instrument_id}_calibration.json"
    path = output_dir / filename
    
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    
    return path


def apply_calibration(wavenumbers: np.ndarray, calibration: Dict[str, float]) -> np.ndarray:
    """Apply calibration to wavenumber array."""
    return wavenumbers * calibration["scale"] + calibration["offset"]


def main():
    parser = argparse.ArgumentParser(description="Calibrate Raman spectrometer")
    parser.add_argument("--reference", "-r", type=Path, required=True, help="Reference spectrum file")
    parser.add_argument("--instrument", "-i", type=str, required=True, help="Instrument serial number")
    parser.add_argument("--output", "-o", type=Path, default=Path("calibration"), help="Output directory")
    args = parser.parse_args()
    
    # Load reference spectrum
    logger.info(f"Loading reference spectrum from {args.reference}")
    wavenumbers, intensities = load_spectrum(args.reference)
    logger.info(f"Loaded spectrum: {len(wavenumbers)} points, range {wavenumbers.min():.0f}-{wavenumbers.max():.0f} cm^-1")
    
    # Find and match peaks
    matched = find_reference_peaks(wavenumbers, intensities)
    logger.info(f"Matched {len(matched)} peaks: {matched}")
    
    if len(matched) < 2:
        logger.error("Insufficient peaks matched for calibration")
        return 1
    
    # Compute calibration
    calibration = compute_calibration(matched)
    
    # Save calibration
    path = save_calibration(calibration, args.instrument, args.output)
    logger.info(f"Saved calibration to {path}")
    
    # Verify
    corrected = apply_calibration(np.array(list(matched.values())), calibration)
    expected = np.array(list(matched.keys()))
    
    print(f"\n✓ Calibration complete for instrument {args.instrument}")
    print(f"  Scale: {calibration['scale']:.6f}")
    print(f"  Offset: {calibration['offset']:.4f} cm^-1")
    print(f"  RMSE: {calibration['rmse']:.4f} cm^-1")
    print(f"  Saved to: {path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
