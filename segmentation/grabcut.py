"""Segmentation interactive classique avec OpenCV/GrabCut."""
from __future__ import annotations
import cv2
import numpy as np


def normalize_rect(rect: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Rectangle (x, y, largeur, hauteur), strictement à l'intérieur de l'image."""
    x, y, w, h = [int(v) for v in rect]
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    # Il faut laisser un peu d'arrière-plan en bordure de l'image.
    x1, y1 = max(1, x), max(1, y)
    x2, y2 = min(width - 1, x + w), min(height - 1, y + h)
    if x2 - x1 < 3 or y2 - y1 < 3:
        raise ValueError('Rectangle trop petit. Sélectionnez un objet plus grand.')
    return x1, y1, x2 - x1, y2 - y1


def segment_grabcut(
    image_rgb: np.ndarray,
    rect: tuple[int, int, int, int],
    foreground_points: list[tuple[int, int]] | None = None,
    background_points: list[tuple[int, int]] | None = None,
    iterations: int = 5,
    brush_radius: int = 7,
) -> np.ndarray:
    """Retourne un masque booléen HxW. Points = indications facultatives de l'utilisateur.

    Une première segmentation se fait par rectangle, puis une seconde par masque
    annoté lorsque l'utilisateur a ajouté des corrections au pinceau.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError('Image RGB attendue (H, W, 3).')
    h, w = image_rgb.shape[:2]
    if min(h, w) < 8:
        raise ValueError('Image trop petite.')
    rect = normalize_rect(rect, w, h)
    image_bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    init_mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(image_bgr, init_mask, rect, bgd_model, fgd_model, max(1, iterations), cv2.GC_INIT_WITH_RECT)
    foreground_points = foreground_points or []
    background_points = background_points or []
    if foreground_points or background_points:
        for x, y in foreground_points:
            cv2.circle(init_mask, (int(x), int(y)), brush_radius, cv2.GC_FGD, -1)
        for x, y in background_points:
            cv2.circle(init_mask, (int(x), int(y)), brush_radius, cv2.GC_BGD, -1)
        # Appliquer le masque avec les contraintes imposées par l'utilisateur.
        bgd_model.fill(0)
        fgd_model.fill(0)
        try:
            cv2.grabCut(image_bgr, init_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        except cv2.error as exc:
            raise ValueError('Corrections incompatibles : ajoutez de l’avant-plan et de l’arrière-plan.') from exc
    return np.isin(init_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)


def transparent_cutout(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Retourne une image RGBA dont le fond est transparent."""
    if image_rgb.shape[:2] != mask.shape[:2]:
        raise ValueError('Masque et image de tailles différentes.')
    alpha = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return np.dstack((np.asarray(image_rgb, dtype=np.uint8), alpha))


def resize_mask(mask: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST)
