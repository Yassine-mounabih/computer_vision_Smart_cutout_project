"""Mesures par rapport à un masque de vérité terrain (ground truth)."""
from __future__ import annotations
import numpy as np


def mask_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    p, g = np.asarray(prediction) > 0, np.asarray(reference) > 0
    if p.shape != g.shape:
        raise ValueError('Masque prédit et masque réel de tailles différentes.')
    intersection = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    denom = int(p.sum() + g.sum())
    return {'IoU': intersection / union if union else 1.0,
            'Dice': 2 * intersection / denom if denom else 1.0}
