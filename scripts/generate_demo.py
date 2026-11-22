"""Génère des images synthétiques ET leurs vrais masques, sans téléchargement.

Images destinées uniquement à tester le pipeline. Pour un U-Net capable de
segmenter de vraies photos, entraîner aussi sur un jeu de données réel annoté.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def make_sample(seed: int, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    n = size
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    # Fond réaliste simplifié : gradient, bruit, motifs discrets.
    bg_color = r.choice([[228, 223, 217], [185, 205, 212], [230, 218, 194],
                         [194, 202, 182], [195, 183, 190], [216, 224, 237]])
    bg = np.array(bg_color, dtype=np.float32)[None, None, :] + 0.12 * (xx[..., None]-n/2) \
         - 0.09 * (yy[..., None]-n/2) + r.normal(0, 3, (n, n, 1))
    if seed % 3 == 0:
        bg += 9*np.sin(xx[..., None]/(n/14.0))
    bg = np.uint8(np.clip(bg, 0, 255))
    mask_img = Image.new('L', (n, n), 0)
    d = ImageDraw.Draw(mask_img)
    shape = seed % 6
    cx = int(n * (0.48 + r.uniform(-0.06, 0.06)))
    cy = int(n * (0.53 + r.uniform(-0.05, 0.05)))
    wd = int(n*r.uniform(0.25, 0.39))
    ht = int(n*r.uniform(0.42, 0.63))
    x1, x2, y1, y2 = cx-wd//2, cx+wd//2, cy-ht//2, cy+ht//2
    if shape == 0:  # vase
        neck = max(8, wd//5)
        d.ellipse((x1, y1+ht//5, x2, y2), fill=255)
        d.polygon([(cx-neck,y1+9),(cx+neck,y1+9),(cx+neck+wd//6,y1+ht//3),
                   (cx-neck-wd//6,y1+ht//3)], fill=255)
        d.rounded_rectangle((cx-neck-4, y1, cx+neck+4, y1+12), 5, fill=255)
    elif shape == 1:  # tasse avec poignée trouée
        d.rounded_rectangle((x1,y1+ht//6,x2,y2), radius=max(5, wd//7),fill=255)
        handle=max(16,wd//3)
        d.ellipse((x2-7,y1+ht//3,x2+handle,y1+ht*3//4), fill=255)
        d.ellipse((x2+1,y1+ht//3+6,x2+handle-6,y1+ht*3//4-6),fill=0)
    elif shape == 2:  # flacon
        d.rounded_rectangle((x1,y1+ht//4,x2,y2), radius=wd//6, fill=255)
        d.rounded_rectangle((cx-wd//5,y1+ht//10,cx+wd//5,y1+ht//3),3,fill=255)
        d.rectangle((cx-wd//4,y1,cx+wd//4,y1+ht//9),fill=255)
    elif shape == 3:  # pomme stylisée
        d.ellipse((x1,y1+ht//6,x2,y2), fill=255)
        d.ellipse((x1-wd//9,y1+ht//4,cx+wd//15,y2-ht//10), fill=255)
        d.ellipse((cx-wd//15,y1+ht//4,x2+wd//9,y2-ht//10), fill=255)
        d.polygon([(cx-2,y1+ht//5),(cx+2,y1+ht//5),(cx+6,y1-5),(cx+2,y1-5)],fill=255)
        d.ellipse((cx+1,y1-ht//14,cx+wd//3,y1+ht//7),fill=255)
    elif shape == 4:  # lampe
        d.polygon([(cx-wd//5,y1),(cx+wd//5,y1),(x2,y1+ht//3),(x1,y1+ht//3)],fill=255)
        d.rounded_rectangle((cx-3,y1+ht//3,cx+3,y2-9),radius=2,fill=255)
        d.ellipse((x1+wd//8,y2-16,x2-wd//8,y2), fill=255)
    else:  # livre incliné
        d.polygon([(x1+wd//5,y1),(x2,y1+ht//14),(x2-wd//5,y2),(x1,y2-ht//12)],fill=255)
    mask = np.asarray(mask_img, dtype=np.uint8)
    fg = np.array(r.choice([[172, 74, 59], [61, 111, 152], [213, 157, 71],
                            [71, 121, 105], [126, 86, 148], [219, 113, 92]]), dtype=np.float32)
    shift = 28*((xx - x1)/max(wd, 1)-0.5)
    sheen = 13*np.cos((xx-cx)/(wd/3+1))
    texture = r.normal(0, 3.5, (n,n))
    front = np.stack([fg[c] + shift*0.6 + sheen + texture for c in range(3)],axis=-1)
    front = np.uint8(np.clip(front,0,255))
    # L'ombre n'est pas dans le masque : elle appartient au fond.
    shadow = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=max(2,n//38)))
    sh = np.asarray(shadow, dtype=np.float32)/255
    sh = np.roll(np.roll(sh, 4, axis=0), 4, axis=1)
    bg = np.uint8(np.clip(bg.astype(np.float32) - sh[...,None]*22,0,255))
    image = np.where(mask[...,None] > 0, front, bg)
    return image.astype(np.uint8), (mask > 0).astype(np.uint8)


def write_demo(output: Path, count: int = 6, size: int = 320):
    images_dir, masks_dir = output/'images', output/'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        # Hors des seeds d'entraînement par défaut = jeu de test indépendant.
        rgb, mask = make_sample(900+i, size)
        name = f'produit_{i+1:02d}.png'
        Image.fromarray(rgb).save(images_dir/name)
        Image.fromarray(mask*255).save(masks_dir/name)
    print(f'{count} images et masques de référence générés dans {output}')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'demo')
    p.add_argument('--count',type=int,default=6)
    args=p.parse_args()
    write_demo(args.output,args.count)


def make_mixed_sample(seed: int, size: int = 128, photos_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Composites synthétiques sur fonds photographiques (masque exact connu).

    NE PAS présenter ces composites comme un jeu de vraies photographies annotées.
    Les fonds proviennent de photos CC0/domaine public dans demo/photos.
    """
    obj, gt = make_sample(seed, size)
    photo_dir = photos_dir or (ROOT/'demo/photos')
    candidates = sorted(photo_dir.glob('*.jpg'))
    if not candidates:
        return obj, gt
    rng = np.random.default_rng(seed+128914)
    path = candidates[int(rng.integers(0,len(candidates)))]
    with Image.open(path) as p:
        base=p.convert('RGB')
        w,h=base.size
        side=int(min(w,h)*rng.uniform(.57,.98))
        left=int(rng.integers(0,max(1,w-side+1)))
        top=int(rng.integers(0,max(1,h-side+1)))
        background=np.asarray(base.crop((left,top,left+side,top+side)).resize((size,size)),dtype=np.uint8)
    if rng.random()<.5:
        background=np.ascontiguousarray(background[:,::-1])
    # Pixels de l'objet synthétique intacts; fond photo varié, GT binaire exact.
    composite=np.where(gt[...,None]>0,obj,background)
    return composite.astype(np.uint8),gt
