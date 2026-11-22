# Projet 10 — GrabCut + U-Net original + U²-Net-p pré-entraîné

## Ce qui est réellement fourni

Ce projet conserve **sans modification** le modèle original `models/unet_demo.pt`.
L'interface propose désormais, dans **Raffinement**, trois options :

- **U-Net original (local)** : le petit U-Net 4 canaux entraîné sur les images synthétiques du projet.
- **U²-Net-p pré-entraîné (fusion)** : un modèle U²-Net-p déjà entraîné sur des objets saillants, puis guidé par le masque GrabCut pour sélectionner l'objet.
- **U²-Net-p pré-entraîné (masque brut)** : le masque U²-Net-p, limité au composant qui chevauche GrabCut, pour observer la différence avec la fusion.

**ATTENTION :** U²-Net-p est une architecture U²-Net (U-Net imbriqué), différente du petit U-Net de notre projet. Le modèle pré-entraîné reçoit une image **RGB seulement** et détecte les objets saillants ; il ne reçoit pas le masque initial comme 4e canal. Le masque GrabCut sert ensuite à choisir le composant et à limiter les modifications. Le réseau n'a pas été réentraîné sur vos images.

## Installer sur Windows (Python 3.14)

Ouvrir PowerShell dans ce dossier :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_pretrained_u2netp.py
.\.venv\Scripts\python.exe app.py
```

Ou utiliser `INSTALLER_MODELE_PREENTRAINE.bat` (installation/dl) puis exécuter `app.py`.

## Important : téléchargement des poids

Les **vrais poids pré-entraînés ne sont pas inclus dans ce ZIP** car le téléchargement externe n'était pas accessible dans l'environnement de création. Le script récupère le checkpoint `u2netp.onnx` de la publication officielle rembg, vérifie son checksum MD5, puis le stocke dans `models/`. Le premier lancement de cette option demande Internet ; les suivants utilisent la copie locale. Si le réseau bloque GitHub, télécharger manuellement le même fichier :

https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx

Puis placer `u2netp.onnx` dans `models/` : le checksum sera vérifié avant utilisation.

## Méthode de test

1. Charger une photo réelle dans la mini-GUI.
2. Cliquer sur l'objet ou tracer un rectangle puis lancer GrabCut.
3. Sélectionner **U-Net original (local)** puis **Affiner avec U-Net** et observer les résultats.
4. Sur **la même image et le même masque GrabCut**, sélectionner **U²-Net-p pré-entraîné (fusion)** puis relancer l'affinage.
5. Tester le **masque brut** pour voir ce que prédit vraiment le réseau pré-entraîné.
6. Si vous disposez d'un masque de référence, calculer IoU et Dice. Sans masque de référence, vous ne pouvez comparer que visuellement.

La présence d'un modèle pré-entraîné **ne garantit pas** une meilleure extraction de tous les objets. Si plusieurs objets existent, U²-Net-p peut privilégier le plus saillant, pas celui choisi par l'utilisateur.

## Sources du modèle et licence

- Architecture U²-Net : https://github.com/xuebinqin/U-2-Net (Apache-2.0)
- Poids ONNX fournis par rembg : https://github.com/danielgatis/rembg (MIT pour rembg)
- `u2netp.onnx` : https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx

Les autres scripts, exemples et tests du ZIP de départ sont conservés. Le modèle original n'est jamais écrasé.
