"""Évaluation honnête, sans utilisation du masque GT durant l'inférence.

Deux expériences DIFFÉRENTES :
1) Entrée normale -> GrabCut seul vs U-Net + fusion.
2) Masque volontairement érodé -> U-Net, puis option distincte
   de re-segmentation GrabCut guidée par l'image. Cette seconde option
   n'est PAS attribuée au Deep Learning.
"""
from __future__ import annotations
from pathlib import Path
import csv
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segmentation.grabcut import segment_grabcut, transparent_cutout
from segmentation.metrics import mask_metrics
from segmentation.refine import refine_with_unet, load_model
from segmentation.repair import recover_with_grabcut


def montage(image, masks, titles, output):
    """Images comparables avec légendes explicites ; noir = transparent."""
    h, w = image.shape[:2]
    strip = Image.new('RGB', (w * len(masks), h + 58), '#f6f7f9')
    for i, (mask, title) in enumerate(zip(masks, titles)):
        if mask is None:
            img = image
        else:
            img = np.where(mask[..., None].astype(bool), image, np.zeros_like(image))
        strip.paste(Image.fromarray(img, 'RGB'), (i*w, 58))
        draw = ImageDraw.Draw(strip)
        draw.text((i*w + 10, 19), title, fill='#132d4e')
    strip.save(output)


def main():
    model, info = load_model(ROOT / 'models/unet_demo.pt')
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    rows, robustness_rows = [], []
    for path in sorted((ROOT / 'demo/images').glob('*.png')):
        img = np.asarray(Image.open(path).convert('RGB'))
        gt = (np.asarray(Image.open(ROOT / 'demo/masks' / path.name).convert('L')) > 127).astype(np.uint8)
        h, w = gt.shape
        # Choisir une boîte fixe SANS regarder le masque de référence.
        rect = (int(w*.12), int(h*.12), int(w*.76), int(h*.76))
        coarse = segment_grabcut(img, rect, iterations=5)
        refined, _ = refine_with_unet(img, coarse, model)
        m1 = mask_metrics(coarse, gt)
        m2 = mask_metrics(refined, gt)
        # Option distincte qui peut améliorer les détails manquants.
        guided, applied = recover_with_grabcut(img, coarse)
        mg = mask_metrics(guided, gt)
        rows.append({'image': path.name, 'IoU_GrabCut':round(m1['IoU'],4),
                     'IoU_GrabCut_UNet':round(m2['IoU'],4),
                     'IoU_GrabCut_guide_OPTION':round(mg['IoU'],4),
                     'Dice_GrabCut':round(m1['Dice'],4),
                     'Dice_GrabCut_UNet':round(m2['Dice'],4),
                     'Dice_GrabCut_guide_OPTION':round(mg['Dice'],4)})
        # Scénario stress : érosion artificielle qui efface des structures minces.
        damaged = cv2.erode(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)), iterations=1)
        unet_repaired, _ = refine_with_unet(img, damaged, model)
        guided_repaired, enabled = recover_with_grabcut(img, damaged)
        bad = mask_metrics(damaged, gt)
        bu = mask_metrics(unet_repaired, gt)
        bg = mask_metrics(guided_repaired, gt)
        robustness_rows.append({'image':path.name,
                                'IoU_masque_erode':round(bad['IoU'],4),
                                'IoU_UNet_apres_erosion':round(bu['IoU'],4),
                                'IoU_GrabCut_guide_apres_erosion_OPTION':round(bg['IoU'],4)})
        montage(img, [None, coarse, refined, guided],
                ['Image originale', 'GrabCut', 'U-Net + fusion', 'GrabCut guidé (option)'],
                output / (path.stem + '_comparaison.png'))
        montage(img, [None, damaged, unet_repaired, guided_repaired],
                ['Image originale', 'Masque erode', 'U-Net sur érosion', 'GrabCut guidé (option)'],
                output / (path.stem + '_robustesse.png'))
        Image.fromarray(transparent_cutout(img, coarse), 'RGBA').save(output / (path.stem + '_grabcut.png'))
        Image.fromarray(transparent_cutout(img, refined), 'RGBA').save(output / (path.stem + '_unet.png'))
        print(f"{path.name}: normal GC={m1['IoU']:.3f}; UNet={m2['IoU']:.3f}; Option GrabCut guidé={mg['IoU']:.3f}; "
              f"érosion={bad['IoU']:.3f}; U-Net={bu['IoU']:.3f}; option guidée={bg['IoU']:.3f}")
    if not rows:
        raise RuntimeError('Aucune image démo. Exécuter scripts/generate_demo.py')
    for filename, arr in [('scores.csv', rows), ('robustesse_scores.csv', robustness_rows)]:
        with (output / filename).open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=arr[0].keys())
            writer.writeheader()
            writer.writerows(arr)
    print('\nMoyennes NORMAL :', {k:round(float(np.mean([r[k] for r in rows])),4)
                                    for k in rows[0] if k!='image'})
    print('Moyennes ÉROSION :', {k:round(float(np.mean([r[k] for r in robustness_rows])),4)
                                    for k in robustness_rows[0] if k!='image'})
    print('NOTE : la réparation guidée est un second GrabCut, PAS une performance attribuable à U-Net.')

if __name__ == '__main__':
    main()
