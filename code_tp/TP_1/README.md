# TP 1 — Inférence avec un ResNet-18 pré-entraîné

## But du TP

Utiliser un réseau déjà entraîné (ResNet-18, ImageNet, 1000 classes) **sans l'entraîner** :
charger le modèle, prétraiter une image, faire le forward et lire la classe prédite.

## Fichier à utiliser

| Fichier | Rôle |
|---------|------|
| `TP_pytorch_1_init.py` | Squelette à compléter (cherche les `TODO`) |

Lancement, depuis la racine du projet (avec le venv du projet) :

```bash
python code_tp/TP_1/TP_pytorch_1_init.py
```

## Images utilisées

`data/test_images/` (sous-dossiers `cat/` et `no_cat/`). Le script traite toutes les images
trouvées et affiche, pour chacune, la classe ImageNet la plus probable.

## Où est sauvegardé le modèle ?

Nulle part : il n'y a pas d'entraînement. Les poids ImageNet sont téléchargés automatiquement
par torchvision au premier lancement (cache `~/.cache/torch`).

## Où sont les résultats ?

Uniquement dans le terminal (shape des tenseurs, classe prédite). Rien n'est écrit dans
`plots/` ni dans `models/`.
