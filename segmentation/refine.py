"""Inférence U-Net et ajustement conservateur du masque GrabCut."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from .unet import UNetRefiner

MODEL_SIZE = 128


def load_model(path: str | Path, device: str = 'cpu') -> tuple[UNetRefiner, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Modèle non trouvé : {path}. Entraînez-le avec scripts/train_unet.py.')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = UNetRefiner(base_channels=int(checkpoint.get('base_channels', 12)))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model, checkpoint


def prepare_tensor(image_rgb: np.ndarray, coarse_mask: np.ndarray, size: int = MODEL_SIZE) -> torch.Tensor:
    img = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mask = cv2.resize(coarse_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    stacked = np.concatenate((img, mask[..., None]), axis=2)
    return torch.from_numpy(np.ascontiguousarray(stacked.transpose(2, 0, 1))).unsqueeze(0)


def refine_with_unet(image_rgb: np.ndarray, coarse_mask: np.ndarray, model: UNetRefiner,
                     device: str = 'cpu', fusion: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (masque affiné, probabilité du premier plan) à la taille initiale.

    Fusion locale : autorise les corrections seulement près des frontières,
    avec seuils U-Net (0,80 / 0,20), limite de changements et cohérence couleur.
    Les couleurs RGB de l'image originale ne sont jamais recréées ni altérées.
    """
    h, w = coarse_mask.shape
    tensor = prepare_tensor(image_rgb, coarse_mask).to(device)
    model.eval()
    with torch.inference_mode():
        prob_small = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    probs = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
    if fusion:
        # Fusion prudente : U-Net ne modifie que les pixels proches des frontières
        # et pour lesquels sa prédiction est suffisamment confiante.
        # Limiter les changements à une bande de quelques pixels de part et
        # d'autre du contour : l'ancien calcul agrandissait beaucoup trop cette
        # bande sur les grandes photographies.
        radius = max(2, min(7, round(min(h, w) * 0.012)))
        side = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        binary = (coarse_mask > 0).astype(np.uint8)
        boundary = (cv2.dilate(binary, kernel) > 0) & (cv2.erode(binary, kernel) == 0)
        additions = boundary & (binary == 0) & (probs > 0.80)
        removals = boundary & (binary == 1) & (probs < 0.20)
        candidate = additions | removals
        # Protection additionnelle : un petit U-Net issu d'un entraînement
        # synthétique ne doit pas déplacer de manière massive les frontières.
        max_changes = max(1, int(min(0.02 * binary.size,
                                     0.12 * max(1, int(binary.sum())))))
        if int(candidate.sum()) > max_changes:
            ys, xs = np.where(candidate)
            confidence = np.where(binary[ys, xs] == 0,
                                  probs[ys, xs] - 0.80, 0.20 - probs[ys, xs])
            chosen = np.argpartition(confidence, -max_changes)[-max_changes:]
            restricted = np.zeros_like(candidate)
            restricted[ys[chosen], xs[chosen]] = True
            additions &= restricted
            removals &= restricted
        # Vérifier la cohérence visuelle locale : un ajout doit davantage
        # ressembler aux pixels proches de l'objet qu'au fond (et inversement).
        # Ce garde-fou ne réentraîne pas U-Net et n'utilise aucun masque réel.
        lab = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        fg = binary.astype(np.float32)
        bg = 1.0 - fg
        sigma = max(2.0, min(h, w) * 0.015)
        blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
        near_fg = blur(fg)
        near_bg = blur(bg)
        fg_color = blur(lab * fg[..., None]) / (near_fg[..., None] + 1.e-3)
        bg_color = blur(lab * bg[..., None]) / (near_bg[..., None] + 1.e-3)
        distance_fg = np.linalg.norm(lab - fg_color, axis=2)
        distance_bg = np.linalg.norm(lab - bg_color, axis=2)
        valid_local = (near_fg > 0.01) & (near_bg > 0.01)
        additions &= valid_local & (distance_fg + 5.0 < distance_bg)
        removals &= valid_local & (distance_bg + 5.0 < distance_fg)
        refined = binary.copy()
        refined[additions] = 1
        refined[removals] = 0
    else:
        refined = (probs >= 0.50).astype(np.uint8)
    return refined.astype(np.uint8), probs
