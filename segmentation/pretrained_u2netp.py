"""U²-Net-p réellement pré-entraîné (salient-object segmentation), format ONNX.

Poids officiels publiés par rembg; téléchargés au premier usage.
Ce modèle n'est PAS le U-Net à 4 canaux entraîné dans ce projet.
Il utilise seulement l'image RGB et ne sait pas choisir une instance sur commande.
Le masque GrabCut guide donc la sélection de l'objet dans l'application.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import os
import tempfile
from urllib.request import Request, urlopen
import cv2
import numpy as np

URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
CHECKSUM_MD5 = "8e83ca70e441ab06c318d82300c84806"  # rembg/sessions/u2netp.py
DEFAULT_PATH = Path(__file__).resolve().parent.parent / 'models' / 'u2netp.onnx'


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(path: str | Path = DEFAULT_PATH, download: bool = True) -> Path:
    path = Path(path)
    if path.exists():
        if _md5(path) == CHECKSUM_MD5:
            return path
        raise ValueError(f'Poids ONNX invalides (somme MD5 différente) : {path}. '
                         'Renommez le fichier puis relancez le téléchargement.')
    if not download:
        raise FileNotFoundError(f'Poids pré-entraînés absents : {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='u2netp_',suffix='.onnx.part',dir=path.parent,delete=False) as tmp:
        temp=Path(tmp.name)
    try:
        req=Request(URL,headers={'User-Agent':'Mozilla/5.0 Segmentation-Project10'})
        with urlopen(req,timeout=50) as response, temp.open('wb') as stream:
            while True:
                part=response.read(1024*1024)
                if not part:break
                stream.write(part)
        if _md5(temp)!=CHECKSUM_MD5:
            raise RuntimeError('Téléchargement U²-Net-p corrompu (checksum MD5 incorrect).')
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)
    return path


class PretrainedU2NetP:
    def __init__(self, path: str | Path = DEFAULT_PATH, download: bool = True):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError('Installez la dépendance : python -m pip install onnxruntime') from e
        weights=ensure_weights(path,download=download)
        self.session=ort.InferenceSession(str(weights),providers=['CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name
        self.output_name=self.session.get_outputs()[0].name

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim!=3 or image_rgb.shape[2]!=3:
            raise ValueError('Image RGB HxWx3 attendue.')
        h,w=image_rgb.shape[:2]
        resized=cv2.resize(image_rgb,(320,320),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        mean=np.array([0.485,0.456,0.406],np.float32)
        std=np.array([0.229,0.224,0.225],np.float32)
        inp=((resized-mean)/std).transpose(2,0,1)[None].astype(np.float32)
        output=self.session.run([self.output_name],{self.input_name:inp})[0]
        output=np.asarray(output,dtype=np.float32).reshape(320,320)
        lo=float(output.min());hi=float(output.max())
        probs=np.zeros_like(output) if hi-lo<1e-7 else (output-lo)/(hi-lo)
        return np.clip(cv2.resize(probs,(w,h),interpolation=cv2.INTER_LINEAR),0,1).astype(np.float32)


def refine_with_pretrained(image_rgb: np.ndarray, coarse_mask: np.ndarray,
                           model: PretrainedU2NetP, *, fusion: bool = True) -> tuple[np.ndarray,np.ndarray]:
    """Guide U²-Net-p avec sélection GrabCut, sans prétendre qu'il s'agit du U-Net d'origine.

    fusion=True : modifications autour du masque GrabCut, sur une bande plus large
    que la fusion du petit modèle, sans expansion globale libre.
    fusion=False : masque U²-Net-p brut limité au composant chevauchant GrabCut.
    """
    coarse=(np.asarray(coarse_mask)>0).astype(np.uint8)
    if coarse.shape!=image_rgb.shape[:2]:
        raise ValueError('Image et masque de tailles incompatibles')
    probs=model.predict(image_rgb)
    if not np.any(coarse):
        return coarse.copy(),probs
    h,w=coarse.shape
    foreground=probs>0.48
    n, labels,stats,_=cv2.connectedComponentsWithStats(foreground.astype(np.uint8),8)
    overlap=[]
    for idx in range(1,n):
        a=int(np.logical_and(labels==idx,coarse>0).sum())
        if a: overlap.append((a,idx))
    if not overlap:
        return coarse.copy(),probs
    object_mask=(labels==max(overlap)[1])
    if not fusion:
        return object_mask.astype(np.uint8),probs

    # Les pixels loin du masque GrabCut ne sont jamais ajoutés.
    radius=max(4,min(32,round(min(h,w)*0.035)))
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    near=cv2.dilate(coarse,k)>0
    # Ajouter les contours très probables du même objet; retirer les pixels
    # sur lesquels le modèle pré-entraîné a une très faible confiance.
    out=coarse.copy()
    add=(coarse==0)&near&object_mask&(probs>0.75)
    remove=(coarse>0)&(probs<0.12)
    # Limiter une suppression massive si le modèle voit autre chose que l'objet.
    if int(remove.sum()) <= int(0.10*max(1,int(coarse.sum()))):
        out[remove]=0
    out[add]=1
    return out.astype(np.uint8),probs
