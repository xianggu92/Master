import os
import random

import numpy as np
import torch


def configure_reproducibility(seed: int) -> int:
    """Configure all RNGs and make PyTorch fail on nondeterministic operations."""
    seed = int(seed)

    # Required by deterministic CUDA matrix multiplications. This must be set
    # before the first CUDA context is created.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return seed


def deterministic_median(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Return median values without CUDA's nondeterministic median indices."""
    size = values.shape[dim]
    if size == 0:
        raise ValueError("cannot compute the median of an empty dimension")

    # torch.median(dim=...) selects the lower middle value for even sizes.
    # Sorting and indexing preserves that behavior without producing indices
    # whose ordering is undefined when multiple values are equal.
    sorted_values = torch.sort(values, dim=dim).values
    index = (size - 1) // 2
    return sorted_values.select(dim, index)
