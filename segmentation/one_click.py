"""Sélection mono-clic guidée par la couleur + GrabCut.

Le clic est un *indice positif*, et non un détecteur d'objets universel.
Une région connectée autour du clic sert à proposer une boîte locale ; GrabCut
optimise ensuite la segmentation. Le résultat est ramené à la composante
connectée qui contient le clic (ou la plus proche).
"""
from __future__ import annotations

import cv2
import numpy as np

from .grabcut import normalize_rect, segment_grabcut


def _seed_region(lab: np.ndarray, click: tuple[int, int], threshold: int) -> np.ndarray:
    h, w = lab.shape[:2]
    mask = np.zeros((h+2, w+2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (1 << 8)
    # Une plage symétrique dans Lab peut bien gérer l'éclairage sur un objet simple.
    cv2.floodFill(lab.copy(), mask, click, None, (threshold,)*3, (threshold,)*3, flags)
    return mask[1:-1, 1:-1].astype(bool)


def propose_click_rect(image_rgb: np.ndarray, click: tuple[int,int],
                       zone_scale: float = 1.0) -> tuple[int,int,int,int]:
    """Boîte initiale estimée à partir de la région similaire et connectée au clic.

    Fallback explicite pour images très texturées/ambigües. La boîte n'utilise
    jamais de masque de vérité terrain.
    """
    img=np.ascontiguousarray(image_rgb,dtype=np.uint8)
    if img.ndim!=3 or img.shape[2]!=3:
        raise ValueError('Image couleur RGB nécessaire.')
    h,w=img.shape[:2];x,y=map(int,click)
    if not (0<=x<w and 0<=y<h):
        raise ValueError('Le clic doit être à l’intérieur de l’image.')
    if not np.isfinite(zone_scale) or not (0.5 <= zone_scale <= 2.0):
        raise ValueError('La taille de la zone clic doit être entre 50 % et 200 %.')
    lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    candidates=[]
    for tol in (9, 15, 22, 30, 41):
        region=_seed_region(lab,(x,y),tol)
        count=int(region.sum())
        frac=count/(w*h)
        if frac<0.0012 or frac>0.5:
            continue
        ys,xs=np.where(region)
        x0,x1=int(xs.min()),int(xs.max());y0,y1=int(ys.min()),int(ys.max())
        bw,bh=x1-x0+1,y1-y0+1
        # Prefer a coherent connected region that does not fill image border.
        touches=int(x0<=1)+int(y0<=1)+int(x1>=w-2)+int(y1>=h-2)
        fill=count/max(1,bw*bh)
        score=(min(frac,0.13)/0.13)*0.55+min(fill,1)*0.45-0.35*touches
        if bw<max(8,w//25) or bh<max(8,h//25):score-=0.2
        candidates.append((score,(x0,y0,x1,y1)))
    if candidates:
        _,(x0,y0,x1,y1)=max(candidates,key=lambda p:p[0])
        bw,bh=x1-x0+1,y1-y0+1
        px=max(9,int(.27*bw),int(w*.04))
        py=max(9,int(.27*bh),int(h*.04))
        x0-=px;y0-=py;x1+=px;y1+=py
        # Too narrow a seed region? Give GrabCut space to optimize.
        if x1-x0<w*.16:
            x0=min(x0,int(x-w*.08));x1=max(x1,int(x+w*.08))
        if y1-y0<h*.16:
            y0=min(y0,int(y-h*.08));y1=max(y1,int(y+h*.08))
    else:
        # No reliable color seed: generalized region centered on chosen object.
        x0=x-round(w*.30);x1=x+round(w*.30)
        y0=y-round(h*.35);y1=y+round(h*.35)
    # On règle l'étendue de la boîte proposée autour du pixel cliqué. À 100 %,
    # on garde la proposition automatique. La zone peut être plus petite ou plus
    # grande, mais son centre d'intérêt (le clic) reste dans le rectangle.
    rect = normalize_rect((x0,y0,x1-x0,y1-y0),w,h)
    if zone_scale == 1.0:
        return rect
    rx,ry,rw,rh=rect
    left=max(1, round(x - (x-rx)*zone_scale))
    top=max(1, round(y - (y-ry)*zone_scale))
    right=min(w-1, round(x + (rx+rw-x)*zone_scale))
    bottom=min(h-1, round(y + (ry+rh-y)*zone_scale))
    # Dimensions minimales raisonnables pour cv2.grabCut.
    if right-left < 8:
        left=max(1,min(left,x-4))
        right=min(w-1,max(right,x+4))
    if bottom-top < 8:
        top=max(1,min(top,y-4))
        bottom=min(h-1,max(bottom,y+4))
    return normalize_rect((left,top,right-left,bottom-top),w,h)


def keep_clicked_component(mask: np.ndarray, click: tuple[int,int]) -> np.ndarray:
    """Évite de détourer simultanément les autres objets séparés du sujet cliqué."""
    binary=(np.asarray(mask)>0).astype(np.uint8)
    if not binary.any():return binary
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary,8)
    if n<=2:return binary
    x,y=map(int,click)
    chosen=int(labels[y,x]) if 0<=x<binary.shape[1] and 0<=y<binary.shape[0] else 0
    if chosen==0:
        # Choix de la composante la plus proche du clic en distance pixel réelle.
        best=float('inf')
        for i in range(1,n):
            xx,yy=np.where(labels==i)[1],np.where(labels==i)[0]
            if xx.size==0:continue
            distance=np.min((xx-x)**2+(yy-y)**2)
            if distance<best:best=distance;chosen=i
    return (labels==chosen).astype(np.uint8)


def segment_one_click(image_rgb: np.ndarray, click: tuple[int,int], iterations: int=5,
                      zone_scale: float = 1.0,
                      foreground_points: list[tuple[int,int]] | None=None,
                      background_points: list[tuple[int,int]] | None=None
                      ) -> tuple[np.ndarray,tuple[int,int,int,int]]:
    """Segmenter à partir d'un clic ; retourne (masque, boîte calculée)."""
    rect=propose_click_rect(image_rgb,click,zone_scale=zone_scale)
    fg=[click]+(foreground_points or [])
    result=segment_grabcut(image_rgb,rect,foreground_points=fg,
                           background_points=background_points,iterations=iterations,
                           brush_radius=max(3,round(min(image_rgb.shape[:2])*.012)))
    result=keep_clicked_component(result,click)
    if int(result.sum())<10:
        raise ValueError('Objet non isolé avec ce clic. Essayez une boîte rectangulaire ou le pinceau.')
    return result,rect
