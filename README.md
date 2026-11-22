# Projet 10 — Outil de découpage intelligent (GrabCut + Deep Learning)

Version **corrigée et renforcée** du projet : une mini-application Tkinter de détourage, en français, où l'utilisateur clique sur un objet ou dessine un rectangle, puis compare **GrabCut seul** à **GrabCut + U-Net + fusion conservatrice**.

> **Limite scientifique importante :** aucun algorithme classique ou petit réseau ne garantit un détourage parfait sur toute photo. Le clic est un *indice interactif* : sur des scènes complexes, il peut falloir un rectangle et des coups de pinceau. Le U-Net fourni est entraîné sur synthétique ; sa supériorité sur les photographies réelles n'est pas démontrée. Les fichiers « robustesse » contiennent des masques volontairement érodés, pas une comparaison GrabCut normale.

## IMPORTANT — images de « robustesse » et réparation des détails (v3)

Les images `_robustesse.png` **commencent volontairement par éroder le masque de GrabCut**. Sur une lampe, l'érosion peut détruire toute la tige : le petit U-Net n'est pas capable de la retrouver à lui seul. Ce ne sont pas des résultats normaux GrabCut vs U-Net.

La v3 ajoute le bouton **« 3. Réparer parties manquantes (GrabCut) »**, qui tente une seconde segmentation guidée par les couleurs et le masque. Sur les démos synthétiques, cette option peut récupérer des formes entières. **C'est une correction GrabCut distincte de U-Net**, à ne pas confondre avec les résultats du Deep Learning. Sur des photos réelles, vérifier visuellement et corriger au pinceau.

**Export** est réglé par défaut sur **GrabCut seul** pour éviter de sauvegarder une version U-Net moins bonne sans le vouloir. Choisir **Résultat de droite** pour exporter U-Net ou la réparation optionnelle.

Lire `CORRECTIONS_V3.md` pour les explications et les expériences corrigées.

## 1. Correspondance exacte avec le sujet du professeur

| Exigence page 10 du PDF « Projets master SIDA » | Comment c'est réalisé |
|---|---|
| « Segmenter un objet d'un clic » | Outil **Un clic** : le clic définit un point positif, on estime une boîte à partir de la couleur et de la connectivité, puis GrabCut ET U-Net se lancent automatiquement. Pour les objets difficiles, la correction manuelle reste disponible. |
| « GrabCut » | `segmentation/grabcut.py` et `segmentation/one_click.py` |
| « Raffinement avec U-Net » | `segmentation/unet.py`, poids `models/unet_demo.pt`, inférence et fusion `segmentation/refine.py` |
| « Interface simple » | `app.py` (Tkinter), trois panneaux, outils de sélection, export |
| « Images variées » | Six objets synthétiques **annotés** et quatre photographies CC0/domaine public ; chargement des photos de l'utilisateur |
| « Application mini-GUI » | Application Python de bureau, sans serveur web |
| « Comparaison GrabCut seul / amélioré » | Trois panneaux, export image de comparaison, IoU/Dice sur données annotées, CSV et images de résultats ; option de réparation guidée supplémentaire |

## 2. Installation et lancement (Windows PowerShell)

**Vous avez Python 3.14 ?** Essayez d'abord cette version : les bibliothèques récentes peuvent fonctionner, mais cette archive a été testée ici avec Python 3.13, et non 3.14 sous Windows. Si une dépendance refuse de s'installer, utilisez Python 3.12 dans un environnement virtuel séparé sans désinstaller 3.14.

Ouvrir PowerShell dans ce dossier, puis exécuter :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

En cas d'échec avec Python 3.14, installez Python 3.12 et remplacez seulement la première ligne par `py -3.12 -m venv .venv`. La commande PowerShell fonctionne **sans activation** de l'environnement virtuel.

Après installation, double-cliquez sur `lancer_windows.bat` ou exécutez la dernière commande ci-dessus.

**Linux :** `python3 -m venv .venv`, `./.venv/bin/pip install -r requirements.txt`, `./.venv/bin/python app.py` (installer `python3-tk` si nécessaire).

**Le modèle U-Net est déjà présent** (`models/unet_demo.pt`) : aucun réentraînement n'est nécessaire pour lancer la GUI. L'installation de PyTorch peut nécessiter une connexion internet et plusieurs centaines de Mo.

## 3. Utilisation complète

### A. Mode « Un clic » — le plus rapide

1. Cliquez **Ouvrir une image** ou choisissez un des exemples dans le menu.
2. Sélectionnez **Un clic (GrabCut + U-Net)**.
3. Cliquez **à l'intérieur de l'objet souhaité**, loin du bord si possible.
4. Le traitement tourne hors du fil de l'interface ; après le calcul, les trois panneaux montrent original, GrabCut et GrabCut + U-Net.
5. Si le détourage n'est pas satisfaisant, corrigez avec le rectangle ou les pinceaux.

**Pourquoi un clic peut échouer :** GrabCut ne reconnaît pas sémantiquement tous les objets ; si la couleur de l'objet est similaire au fond ou si l'objet touche le cadre, la boîte estimée peut être incomplète. Le mode rectangle est alors recommandé.

### B. Mode « Rectangle » — plus fiable sur les photos difficiles

1. Choisissez **Rectangle**.
2. Dessinez un rectangle encadrant tout l'objet, en laissant du fond autour.
3. Cliquez **1. Lancer GrabCut**.
4. Examinez la segmentation et cliquez **2. Affiner avec U-Net**.
5. Si nécessaire, passez aux pinceaux pour **garder** (vert) ou **supprimer** (rouge) des régions, puis **relancez GrabCut et U-Net**.
6. **Annuler pinceau** permet d'annuler le dernier trait ; `Rayon` ajuste la taille du pinceau.

### C. Sauvegarder et comparer

- **Enregistrer PNG** : crée un fichier RGBA transparent, sans changer les couleurs RGB originales de l'objet.
- **Export : GrabCut seul / Résultat de droite** : choisissez lequel enregistrer ; le réseau peut parfois être moins bon, et la troisième étape est une option GrabCut guidée.
- **Flou alpha** de 0 à 3 pixels : option visuelle pour adoucir le bord au moment de l'export ; aucun réentraînement.
- **Exporter comparaison** : image à trois colonnes original, GrabCut et GrabCut + U-Net.
- **Charger masque de référence** : charge un masque binaire annoté pour calculer IoU et Dice sur votre photo (blanc = objet, noir = fond).
- **Voir les contours** : superpose le contour GrabCut sur le panneau gauche.

### D. Les photos fournies

- `demo/images/` + `demo/masks/` : six images **synthétiques** et leur **vérité terrain**. IoU/Dice s'affichent dans la GUI.
- `demo/photos/` : quatre **photographies réelles** ; *pas* de vérité terrain fournie, donc **pas d'IoU/Dice valide** sans annotation indépendante.
- Provenance/licences des photos : `demo/photos/ATTRIBUTION.txt`.

## 4. Fonctionnement technique

```text
Image RGB + clic (ou rectangle)
          ↓
Segmentation interactive par GrabCut (modèles de couleurs GMM + Graph Cut)
          ↓
Masque initial (0 = fond, 1 = objet)
          ↓
Concaténation image RGB (3 canaux) + masque GrabCut (1 canal)
          ↓
U-Net encodeur-décodeur CNN avec skip connections
          ↓
Carte de probabilités d'appartenance à l'objet (0…1)
          ↓
Fusion conservatrice : seuils (0,80/0,20), bande de contour,
limite du nombre de modifications et cohérence locale des couleurs
          ↓
Masque final → canal alpha → PNG RGBA
```

L'**espace latent** est la famille des représentations que peut produire l'encodeur ; à l'exécution le **tenseur latent** se trouve temporairement en RAM/VRAM. Les **skip connections** transmettent des *tenseurs de caractéristiques intermédiaires* de l'encodeur vers le décodeur. Elles ne sont pas la représentation latente finale.

Le **modèle U-Net** ne recrée ni la texture ni les couleurs RGB. Il prédit **un masque**. `transparent_cutout()` concatène les trois canaux RGB originaux avec un alpha `255` sur l'objet et `0` sur le fond.

### Pourquoi la fusion conservatrice ?

Le modèle étant petit et entraîné sur synthétique, ses probabilités peuvent être erronées sur de vraies photos. Le post-traitement ne change des décisions de GrabCut que :

1. dans une **bande autour du contour** ;
2. lorsque U-Net est assez confiant (`>0,80` pour ajouter, `<0,20` pour retirer) ;
3. lorsque la **couleur locale** est plus compatible avec le groupe de destination qu'avec l'autre ;
4. avec un plafond de modifications (au maximum 2 % des pixels de l'image ou 12 % du masque initial).

L'étape corrige **les valeurs du masque**, pas les valeurs RGB des pixels. Elle intervient **après l'inférence U-Net**, avant la génération du PNG.

## 5. Vérifications et résultats observés

Commandes depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/evaluate_demo.py
.\.venv\Scripts\python.exe scripts/evaluate_photos.py
```

**18 tests automatiques réussis** dans l'environnement de vérification, et test Tkinter de bout en bout réussi dans un bureau virtuel : clic, GrabCut, U-Net, export RGBA. Voir `scripts/smoke_gui.py` (nécessite un bureau graphique).

Résultats IoU moyens des **six images synthétiques annotées** avec la fusion renforcée :

| Protocole | IoU moyenne |
|---|---:|
| GrabCut seul, sans perturbation | 0,9733 |
| GrabCut + U-Net + fusion, sans perturbation | 0,9740 |
| Masque initial **artificiellement érodé** (étude distincte) | 0,8074 |
| Masque érodé + U-Net + fusion | 0,8947 |

**Interprétation :** avec une **boîte fixe indépendante du masque de référence**, GrabCut est parfait sur cinq images. Sur la sixième, l'IoU passe de **0,8399 à 0,8440** après U-Net : le gain moyen n'est que **+0,0007**. Les masques volontairement érodés constituent un **autre protocole** ; il serait incorrect de présenter leur IoU 0,8074 comme un résultat normal de GrabCut ou de prétendre que U-Net améliore systématiquement les photographies réelles.

Les quatre résultats sur photographies sont dans `results/photos_qualitatives/`. Ils sont **qualitatifs seulement** et montrent aussi des cas imparfaits. Une évaluation chiffrée sur données réelles nécessite des masques annotés indépendants.

## 6. Entraîner sur ses données annotées

Pour un U-Net capable de mieux généraliser, utiliser un vrai jeu de photographies **avec annotations** (image et masque binaire portant le même nom de base) :

```text
data/
  images/objet_001.jpg
  images/objet_002.png
  masks/objet_001.png
  masks/objet_002.png
```

```powershell
.\.venv\Scripts\python.exe scripts/train_unet.py --images data/images --masks data/masks --epochs 20 --output models/unet_demo.pt
```

Pour réentraîner un modèle de démonstration **sur données synthétiques seulement** :

```powershell
.\.venv\Scripts\python.exe scripts/train_unet.py --epochs 12 --num-demo 180
```

Ou sur **composites synthétiques avec fonds photographiques CC0**, non équivalents à de vraies annotations :

```powershell
.\.venv\Scripts\python.exe scripts/train_unet.py --epochs 12 --num-demo 180 --mixed-backgrounds --output models/unet_mixed.pt
```

Le checkpoint actif livré (`models/unet_demo.pt`) correspond à un entraînement **synthétique de six époques**. L'option `--mixed-backgrounds` est fournie comme expérimentation facultative ; elle ne modifie pas le modèle livré et n'est pas nécessaire au démarrage.

## 7. Structure des fichiers

```text
app.py                           Interface Tkinter, clic et pinceaux, export
lancer_windows.bat               Lanceur Windows
requirements.txt                 Dépendances Python
segmentation/grabcut.py          GrabCut rectangle, annotations, alpha PNG
segmentation/one_click.py        Boîte à partir du clic + composante connexe
segmentation/unet.py             Modèle U-Net CNN à 4 canaux
segmentation/refine.py           Inférence, fusion locale et précautions
segmentation/repair.py           Seconde passe GrabCut guidée optionnelle
segmentation/metrics.py          IoU et Dice
models/unet_demo.pt              Modèle fourni, entraîné sur synthétique
models/unet_experimental_mixed.pt Modèle expérimental (non retenu)
scripts/generate_demo.py         Images synthétiques et composites à fonds photos
scripts/train_unet.py            Entraînement supervisé, support données réelles
scripts/evaluate_demo.py        Évaluation quantitative sur les 6 démos
scripts/evaluate_photos.py      Comparaison visuelle des vraies photos
scripts/smoke_gui.py             Test de l'interface (avec affichage graphique)
demo/images, demo/masks          Images annotées synthétiques
 demo/photos/                    Quatre photographies CC0/domaine public
results/scores.csv              Évaluation synthétique non perturbée
results/robustesse_scores.csv    Essai séparé : érosion artificielle
results/photos_qualitatives/    Qualitatif, sans IoU/Dice
rapport_projet.md                Rapport prêt pour la soutenance
apercu_interface_v2.png         Capture de l'application corrigée
tests/                           Tests Pytest
```

## 8. Ce qui n'est PAS garanti

- Un unique clic ne suffit pas forcément pour détourer les cheveux, la fourrure, le verre, un objet tronqué, ou une couleur très proche du fond.
- Le modèle synthétique livré n'est pas un modèle universel de suppression d'arrière-plan.
- Une modification faible du masque n'est **pas nécessairement une amélioration** ; toujours comparer visuellement ou avec un masque indépendant.
- Le code fonctionne et a été testé sous Linux avec affichage virtuel ; l'installation exacte et les performances peuvent dépendre de Python, Windows, pilotes, ressources matérielles et photos choisies.

Ce sont des limites de la méthode, pas une absence des éléments demandés dans l'énoncé.

## 9. Mesures reproductibles et limites de la v3

Sur les **6 images synthétiques annotées**, avec une boîte de départ définie sans regarder le masque de référence :

| Approche normale | IoU moyenne |
|---|---:|
| GrabCut seul | 0,9733 |
| GrabCut + U-Net + fusion | 0,9740 |
| Réparation optionnelle par second GrabCut guidé | 1,0000 |

Sur les **masques volontairement érodés** (expérience distincte) :

| Approche sur masque dégradé | IoU moyenne |
|---|---:|
| Masque érodé | 0,8074 |
| U-Net + fusion | 0,8947 |
| Réparation optionnelle par second GrabCut guidé | 1,0000 |

Le score 1,0000 de la troisième approche est obtenu **uniquement sur ces six images synthétiques**. Il ne représente **pas** une performance U-Net et ne prédit pas la qualité sur les photos réelles.

Un entraînement expérimental supplémentaire avec 220 composites sur des fonds photographiques et 16 époques a donné une meilleure perte de validation, mais une **IoU de robustesse inférieure** à celle du modèle initial (environ 0,8707 contre 0,8947). Le modèle initial est donc conservé par défaut dans `models/unet_demo.pt`. Les poids expérimentaux sont fournis à part dans `models/unet_experimental_mixed.pt` uniquement pour étude, pas comme amélioration vérifiée.

**Tests de cette version :** 18 tests automatiques réussis et test GUI virtuel réussi (clic, export PNG et bouton de réparation). Vérifiés dans un environnement Python 3.13 ; revérifier sous Python 3.14 sur Windows.


## V4 — Réglage du clic rapide

Le nouveau curseur **Taille zone clic** contrôle l'étendue du rectangle de recherche calculé autour du point cliqué (50–200 %, valeur initiale 100 %). Un rectangle jaune visualise la zone. Après ajustement du curseur, cliquer sur **Relancer ce clic** recalcule GrabCut et U-Net. Le réglage **Rayon** agit uniquement sur les pinceaux de correction, indépendamment du clic rapide. Une zone plus grande ne garantit pas une meilleure segmentation et peut inclure davantage de fond.
