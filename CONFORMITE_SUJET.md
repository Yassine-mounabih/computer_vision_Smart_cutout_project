# Vérification du sujet Projet 10 — PDF « Projets master SIDA » (page 10)

Le sujet original demande :

> Contexte réel : outil type Photoshop, découpage d'objets (pub, e-commerce).
>
> Problématique : segmenter un objet d'un clic.
>
> Objectif pédagogique : comparer segmentation interactive + DL.
>
> Méthodes : GrabCut, raffinement avec U-Net, interface simple.
>
> Données : images variées.
>
> Livrables : application mini-GUI et comparaison GrabCut seul / amélioré.

## Tableau de conformité et preuves

| Exigence | Réalisation vérifiée | Fichiers pertinents |
|---|---|---|
| Découpage d'objet | Masque binaire puis PNG RGBA alpha | `segmentation/grabcut.py`, `app.py` |
| Un clic | Point d'avant-plan, détection d'une région locale, GrabCut et U-Net auto | `segmentation/one_click.py`, `app.py` |
| Segmentation interactive | Boîte, pinceau avant-plan et arrière-plan, annulation | `app.py` |
| GrabCut | GMM + Graph Cut via OpenCV | `segmentation/grabcut.py` |
| Deep Learning | U-Net encodeur-décodeur, 4 canaux RGB+masque | `segmentation/unet.py` |
| Raffinement | U-Net puis fusion locale prudente | `segmentation/refine.py` |
| Interface simple | GUI Tkinter 3 panneaux, sélection et export | `app.py`, `apercu_interface.png` |
| Données variées | 6 objets synthétiques annotés, 4 photos réelles et ajout libre | `demo/` |
| Comparaison | Visuelle et IoU/Dice si annotation réelle | `scripts/evaluate_demo.py`, `scripts/evaluate_photos.py`, `results/` |
| Livrable mini-GUI | Exécutable localement par commande Python | `lancer_windows.bat`, `README.md` |

## Vérifications réalisées dans l'environnement de développement

- `python -m pytest -q` : **15 tests réussis**.
- `xvfb-run -a python scripts/smoke_gui.py` : **réussi** (Tkinter, clic, GrabCut, U-Net, export PNG transparent et photo réelle).
- `python scripts/evaluate_demo.py` : **réussi**.
- `python scripts/evaluate_photos.py` : **réussi**.

**Résultat scientifique :** six masques synthétiques annotés avec une boîte initiale fixe sans utiliser le masque GT : IoU moyenne GrabCut=**0,9733**, IoU moyenne GrabCut + U-Net=**0,9740**. Étude distincte d'érosion : **0,8074 → 0,8947**. Pas de mesure de performance sur les vraies photos sans annotation.

**Limite importante :** « un clic » est bien implémenté comme interaction fonctionnelle, mais ne garantit pas que chaque objet complexe sera correctement détouré. Une qualité comparable à un modèle de segmentation interactif généraliste demanderait un modèle et des données plus riches. Les corrections manuelles rendent la version pédagogique utilisable sur plus de cas.


### Rectification v3 — preuves et limites

La correction des fins détails absents peut nécessiter **une seconde passe GrabCut guidée**, distincte de U-Net. Les images de robustesse utilisent une érosion artificielle, documentée dans `CORRECTIONS_V3.md`. La version corrigée réussit 18 tests automatisés et un test GUI avec export RGBA et réparation de la lampe synthétique. L'objectif « segmenter d'un clic » est implémenté comme une sélection heuristique guidée, et non comme un détecteur universel d'objets. Aucune garantie de segmentation parfaite sur les photos réelles.
