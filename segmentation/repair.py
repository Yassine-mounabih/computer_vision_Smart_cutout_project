"""Option de réparation guidée par l'image, distincte de l'inférence U-Net.

Une re-segmentation GrabCut utilise le masque initial comme amorce de premier plan
et autorise le fond « probable » à redevenir premier plan. Elle peut récupérer
des structures fines détruites par érosion. Ce n'est PAS une prédiction U-Net.

Aucun raffinement ne peut garantir qu'il améliore une image sans vérité terrain.
"""
from __future__ import annotations
import cv2
import numpy as np


def recover_with_grabcut(image_rgb: np.ndarray, mask: np.ndarray,
                         iterations: int = 5, max_area_factor: float = 3.0,
                         min_color_separation: float = 18.0) -> tuple[np.ndarray, bool]:
    """(masque, appliqué). Réparation optionnelle ; jamais automatique.

    Cette procédure préserve les pixels du masque fourni (premier plan certain).
    Si l'objet est trop proche du fond en couleur ou si la proposition grossit
    excessivement, elle renvoie le masque inchangé.
    """
    image = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    coarse = (np.asarray(mask) > 0).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != coarse.shape:
        raise ValueError('Image RGB et masque de même taille requis.')
    h, w = coarse.shape
    if min(h, w) < 16 or coarse.sum() < 30 or coarse.sum() > .85 * coarse.size:
        return coarse, False

    # Un contrôle de séparabilité réduit l'expansion abusive pour les scènes
    # complexes ; il ne prétend pas évaluer la qualité réelle de la prédiction.
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    sure_bg = np.zeros_like(coarse, bool)
    border = max(2, int(min(h, w) * .015))
    sure_bg[:border, :] = True
    sure_bg[-border:, :] = True
    sure_bg[:, :border] = True
    sure_bg[:, -border:] = True
    sure_bg &= (coarse == 0)
    if not sure_bg.any():
        return coarse, False
    fg_sample = lab[coarse != 0]
    bg_sample = lab[sure_bg]
    fg_color = np.median(fg_sample, axis=0)
    bg_color = np.median(bg_sample, axis=0)
    if float(np.linalg.norm(fg_color - bg_color)) < min_color_separation:
        return coarse, False

    # L'extérieur n'est que « probablement » fond : GrabCut peut récupérer
    # des objets manquants, comme une tige fine invisible dans le masque érodé.
    gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc[sure_bg] = cv2.GC_BGD
    gc[coarse != 0] = cv2.GC_FGD
    # Permettre à l'algorithme d'examiner la frontière proche de l'ancien objet.
    ring = (cv2.dilate(coarse, np.ones((3, 3), np.uint8)) > 0) & (coarse == 0) & ~sure_bg
    gc[ring] = cv2.GC_PR_FGD
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), gc, None,
                    bg_model, fg_model, max(1, int(iterations)), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return coarse, False

    candidate = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    # Les parties nouvelles doivent toucher une région du masque de départ.
    num, components, _, _ = cv2.connectedComponentsWithStats(candidate, 8)
    ids = np.unique(components[coarse > 0])
    ids = ids[ids != 0]
    candidate = np.isin(components, ids).astype(np.uint8)
    if num < 2 or candidate.sum() < coarse.sum() or candidate.sum() > max_area_factor * coarse.sum():
        return coarse, False
    # Éviter que de grands morceaux de fond, non justifiés, soient ajoutés.
    if candidate.sum() > .72 * candidate.size:
        return coarse, False
    return candidate, bool(np.any(candidate != coarse))
