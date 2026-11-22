"""Télécharger les poids officiels U²-Net-p (environ 5 Mo) et vérifier le checksum."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segmentation.pretrained_u2netp import ensure_weights
if __name__=='__main__':
    try:
        path=ensure_weights()
        print('Modèle pré-entraîné disponible :',path)
    except Exception as exc:
        print('ERREUR :',exc)
        raise SystemExit(1)
