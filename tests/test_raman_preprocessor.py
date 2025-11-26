import numpy as np
from bentoml_service.pmm_brain import RamanPreprocessor

def test_als_baseline_on_synthetic_signal():
    rp = RamanPreprocessor()

    x = np.linspace(0, 1, 256)
    baseline_true = 0.1 + 0.2 * x
    peaks = np.zeros_like(x)
    peaks[80] += 1.0
    peaks[150] += 0.7

    y = baseline_true + peaks + 0.02 * np.random.randn(256)
    z = rp._als_baseline(y, lam=1e5, p=0.01, niter=10)

    # Residual should be approx baseline+peaks - baseline ≈ peaks
    residual = y - z

    # Baseline approx: z should correlate strongly with baseline_true
    corr = np.corrcoef(z, baseline_true)[0, 1]
    assert corr > 0.9

    # Residual should have clear peaks at the right indices
    assert residual[80] > 0.5
    assert residual[150] > 0.3
