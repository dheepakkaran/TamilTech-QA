"""Deterministic seeding across stdlib, numpy, torch, and transformers."""

from __future__ import annotations

import os
import random
from typing import Optional


def seed_everything(seed: int = 42, deterministic_cuda: bool = False) -> int:
    """Seed all relevant RNGs for reproducibility.

    Seeds the Python ``random`` module, ``numpy``, ``torch`` (CPU and CUDA),
    and ``transformers`` (via ``set_seed`` if available). Sets the
    ``PYTHONHASHSEED`` env var.

    Args:
        seed: Integer seed.
        deterministic_cuda: If True, sets ``torch.backends.cudnn.deterministic``
            and disables benchmarking. Slower but bit-exact reproducible.

    Returns:
        The seed that was set.

    Example:
        >>> seed_everything(123)
        123
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass

    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:  # pragma: no cover
        pass

    return seed


def get_seed_from_env(default: Optional[int] = 42) -> int:
    """Return ``TAMILTECH_SEED`` from env, falling back to ``default``.

    Args:
        default: Fallback seed value.

    Returns:
        Integer seed.
    """
    raw = os.environ.get("TAMILTECH_SEED")
    return int(raw) if raw else (default if default is not None else 42)
