"""Tests des limites de la restauration guidée et de la documentation honnête."""
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segmentation.repair import recover_with_grabcut
from segmentation.metrics import mask_metrics
from segmentation.grabcut import segment_grabcut

ROOT=Path(__file__).resolve().parents[1]


def test_recover_eroded_lamp_stem():
    img = np.asarray(Image.open(ROOT/'demo/images/produit_05.png').convert('RGB'))
    gt = (np.asarray(Image.open(ROOT/'demo/masks/produit_05.png').convert('L')) > 127).astype(np.uint8)
    coarse = segment_grabcut(img, (38, 38, 243, 243))
    damaged = cv2.erode(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
    repaired, applied = recover_with_grabcut(img, damaged)
    assert applied
    assert mask_metrics(repaired, gt)['IoU'] > .96
    assert mask_metrics(repaired, gt)['IoU'] > mask_metrics(damaged, gt)['IoU']


def test_unseparable_image_keeps_mask():
    img = np.full((96,96,3), 128, np.uint8)
    mask = np.zeros((96,96),np.uint8)
    mask[34:62,34:62] = 1
    result, applied = recover_with_grabcut(img, mask)
    assert not applied
    assert np.array_equal(result,mask)


def test_invalid_shapes_raise():
    import pytest
    with pytest.raises(ValueError):
        recover_with_grabcut(np.zeros((32,32,3),np.uint8),np.zeros((20,20),np.uint8))
