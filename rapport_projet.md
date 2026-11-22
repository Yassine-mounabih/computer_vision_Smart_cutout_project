# Rapport technique et bilan — Projet 10

## 1. Objectif et exigences

Le professeur demande un outil type Photoshop permettant de « segmenter un objet d'un clic », en combinant **GrabCut**, **raffinement U-Net** et **mini-interface graphique**, avec des **images variées** et une **comparaison GrabCut seul / amélioré**.

## 2. Architecture fonctionnelle

1. **Acquisition** : l'utilisateur charge une image. L'application travaille sur une copie redimensionnée (max. 900 pixels sur le grand côté), en conservant l'original pour l'export.
2. **Interaction** : un clic initialise le sujet souhaité ; une recherche de région connectée de couleur voisine estime un rectangle. Un rectangle explicite est également disponible.
3. **GrabCut** : à partir d'un rectangle et d'indications avant-/arrière-plan, OpenCV utilise modèles GMM et optimisation Graph Cut pour estimer un masque binaire.
4. **U-Net** : entrée à quatre canaux (RGB et masque GrabCut) ; l'encodeur construit un tenseur latent, le décodeur le réutilise avec les *skip connections* ; prédiction d'une carte de probabilités.
5. **Fusion conservatrice** : modification de certains pixels proches des contours selon la confiance du modèle, la cohérence des couleurs locales et un plafond de modifications ; aucun RGB n'est inventé.
6. **Sortie** : le masque final définit le canal alpha du PNG. Les pixels conservés gardent leurs couleurs originales.

## 3. Technologies, justification

- **Python** : orchestration et écosystème scientifique.
- **OpenCV** : GrabCut, connectivité, morphologie et couleurs Lab.
- **NumPy** : opérations vectorisées sur pixels et masques.
- **PyTorch** : modèle U-Net et inférence du checkpoint.
- **Pillow** : lecture/orientation, dessins de la GUI, PNG RGBA.
- **Tkinter + ttk** : GUI locale, clic, rectangle, pinceaux, progression.
- **Pytest** : vérifications automatisées.

## 4. Détails U-Net et apprentissage

Architecture encodeur-décodeur à trois niveaux, avec *skip connections* et convolutions, réduction par MaxPooling et reconstruction par interpolation. Entrée `(B, 4, 128, 128)` ; sortie `(B, 1, 128, 128)` en logits, convertis en probabilités par Sigmoid. Les 4 canaux d'entrée sont les 3 couleurs RGB normalisées et le masque binaire GrabCut.

Le checkpoint livré a été entraîné **sur 150 objets synthétiques pendant 6 époques**, avec validation distincte ; il ne prétend pas généraliser à tout objet réel. Optimiseur Adam, perte BCE + Dice. Une option existe pour réentraîner avec des **vraies photos annotées** : `scripts/train_unet.py --images ... --masks ...`. Les composites sur fonds photos sont un complément pédagogique, **pas une vérité terrain naturelle**.

## 5. Fusion après U-Net

1. Repérer une bande autour de la frontière GrabCut, de quelques pixels (dilatation/érosion).
2. Proposer un **ajout** si GrabCut=0 et probabilité U-Net > 0,80 ; proposer un **retrait** si GrabCut=1 et probabilité < 0,20.
3. Limiter les corrections à 2 % de l'image ou 12 % de la surface du masque GrabCut.
4. Vérifier la cohérence des couleurs Lab locales : ajouter du côté plus proche de l'objet, retirer du côté plus proche du fond.
5. Produire le masque corrigé. Ce mécanisme est une stratégie algorithmique de **post-traitement**, pas un second modèle entraîné.

## 6. Évaluation

Sur 6 images **synthétiques annotées**, avec la version corrigée :

| Protocole | IoU moyenne |
|---|---:|
| GrabCut seul | **0,9733** |
| GrabCut + U-Net + fusion | **0,9740** |
| Masque GrabCut artificiellement érodé | **0,8074** |
| Masque érodé + U-Net + fusion | **0,8947** |

**Protocole honnête :** l’évaluation utilise une boîte centrale **fixe, définie sans connaître le masque de référence** ; le masque correct est utilisé seulement pour calculer IoU/Dice. Cinq images sur six sont déjà parfaites avec GrabCut ; sur la sixième, l’IoU passe de **0,8399 à 0,8440** avec U-Net/fusion, soit un **gain moyen minime (+0,0007)**.

Fichiers : `results/scores.csv`, `results/robustesse_scores.csv`. Les deux dernières lignes correspondent à un **autre protocole**, avec érosion artificielle. Ne jamais confondre « masque érodé » et GrabCut normal.

Sur quatre **photographies réelles** (chat, tasse, astronaute, fusée), les captures de comparaison sont fournies dans `results/photos_qualitatives/`. **Aucun masque de référence n'est disponible**, donc pas de métriques IoU/Dice légitimes sur ces exemples. Les extractions ne sont pas toutes parfaites et certaines nécessitent correction au pinceau ou rectangle.

## 7. Tests et qualité

- **15 tests Pytest réussis** : validité GrabCut, masque alpha/RGB, métriques, formes U-Net, normalisation rectangle, mono-clic sur les 6 objets, composantes connexes, clic hors image, checkpoint et limite de fusion.
- **Test GUI d'intégration réussi** sous Xvfb : démarrage Tkinter, mono-clic image annotée, calcul de GrabCut et U-Net, export RGBA, chargement photo réelle, nouveau mono-clic.
- Aucun test automatisé ne constitue une garantie de bonne segmentation sur toutes les scènes réelles.

## 8. Ce qui a été corrigé par rapport à la première version

- Mono-clic : l'ancien rectangle était fixé à une proportion constante ; la nouvelle boîte dépend de la couleur et de la région connexe autour du clic, puis le pipeline GrabCut + U-Net se lance automatiquement.
- Interface : navigation entre objets synthétiques et photos réelles, indications et annulation pinceaux, export séparé du masque GrabCut/affiné, export comparaison, possibilité d'ajouter une annotation réelle.
- Fusion : l'ancienne bande s'élargissait excessivement sur photos ; la nouvelle bande est courte, avec contraintes de couleur et plafond de correction.
- Qualité : ajout de 10 tests, test GUI, 4 photos aux licences explicites et comparaisons qualitatives indépendantes des scores synthétiques. L’évaluation n’utilise plus une boîte calculée à partir de la vérité terrain.

## 9. Limites et suites recommandées

Le sujet est **implémenté dans une version pédagogique fonctionnelle**, mais une qualité professionnelle et fiable du détourage « d'un clic » nécessite un modèle de segmentation interactif plus puissant et un ensemble de photos annotées pour évaluer le taux d'erreurs. Pour une soutenance, montrer au moins un cas synthétique, une photo réelle facile, une scène difficile avec corrections, et reconnaître l'absence de preuve de gain du U-Net sur de vraies photos.


## Addendum v3 : limites identifiées dans la soutenance

Les images `*_robustesse.png` appliquent **intentionnellement une érosion du masque de GrabCut**. Cela peut faire disparaître complètement une tige fine. Le U-Net initial ne la reconstruit pas de façon fiable, même si sa IoU sur les bords restants augmente. Une amélioration numérique ne garantit pas un détourage visuellement satisfaisant.

Pour distinguer correctement les méthodes, une **troisième option indépendante** est ajoutée : une seconde segmentation GrabCut guidée par le masque, avec garde-fous de couleur et surface. Cette opération reconstruit la lampe synthétique (IoU de 0,840 à 1,000 sur l'exemple), mais **ne doit jamais être attribuée à U-Net**.

Les tableaux finaux exacts sont dans `results/scores.csv` et `results/robustesse_scores.csv`. Sur six exemples annotés, IoU moyenne : GrabCut 0,9733, GrabCut + U-Net 0,9740, option GrabCut guidé 1,0000 ; sur masque artificiellement érodé : 0,8074, 0,8947 et 1,0000 respectivement. Le test sur quatre photos réelles reste uniquement visuel, sans masque de vérité terrain.

La GUI exporte **GrabCut seul par défaut**. Le choix du résultat droit reste explicite, car U-Net ne garantit pas un gain. En cas de mauvaise segmentation, utiliser les indications au pinceau puis relancer GrabCut.
