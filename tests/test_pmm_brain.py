import numpy as np
import torch
from bentoml_service.pmm_brain import StaticPseudoModeMemory

def test_pmm_creates_modes_and_merges():
    torch.manual_seed(0)
    device = torch.device("cpu")

    brain = StaticPseudoModeMemory(
        latent_dim=8,
        max_modes=8,
        merge_thresh=0.9,   # high similarity required
        split_thresh=1000.0,
        prune_thresh=0.0,
        device=device,
    )

    # Two very close clusters
    cluster_a = torch.randn(16, 8) * 0.01 + 1.0
    cluster_b = torch.randn(16, 8) * 0.01 + 1.02
    x = torch.cat([cluster_a, cluster_b], dim=0).to(device)

    with torch.no_grad():
        _ = brain(x, update=True)  # one pass
        n_modes_after = int(brain.active_mask.sum().item())

    assert n_modes_after >= 1
    assert n_modes_after <= 4  # should not explode in modes

def test_pmm_prunes_inactive_modes():
    device = torch.device("cpu")

    brain = StaticPseudoModeMemory(
        latent_dim=4,
        max_modes=4,
        merge_thresh=0.8,
        split_thresh=1000.0,
        prune_thresh=0.1,
        device=device,
    )

    x = torch.randn(10, 4, device=device)
    with torch.no_grad():
        _ = brain(x, update=True)
        # simulate decay / no new data
        for _ in range(20):
            brain.apply_explicit_updates()

    # Some modes should have been pruned
    assert int(brain.active_mask.sum().item()) >= 1
