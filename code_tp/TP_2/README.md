# TP 2 — Fine-tuning : chat / pas chat

## But du TP

Adapter un ResNet-18 pré-entraîné à un nouveau problème à 2 classes (chat / pas chat) :
remplacer la dernière couche par 2 sorties, entraîner uniquement cette couche
(transfer learning) et suivre les courbes de loss et d'accuracy.

## Fichier à utiliser

| Fichier | Rôle |
|---------|------|
| `TP_pytorch_2_training.py` | Squelette d'entraînement à compléter (cherche les `TODO`) |

Lancement, depuis la racine du projet (avec le venv du projet) :

```bash
python code_tp/TP_2/TP_pytorch_2_training.py
```

Pour tester le modèle entraîné, utiliser ensuite les scripts de [`../utils/`](../utils/README.md).

## Images utilisées

- `images/cat/` et `images/no_cat/` : entraînement et validation (découpage automatique).
- `data/test_images/` : images de test, utilisées par les scripts d'évaluation (pas pendant l'entraînement).

## Où est sauvegardé le modèle ?

Dans `models/` (dossier créé au premier lancement, non versionné) :

| Fichier | Contenu |
|---------|---------|
| `models/cat_model_init.pth` | Modèle à la **dernière** epoch |
| `models/cat_model_best.pth` | Modèle de l'epoch avec la **meilleure accuracy de validation** |

Chaque lancement **écrase** ces deux fichiers.

## Où sont les résultats ?

- **Courbes** : `plots/training_accuracy.png` et `plots/training_loss.png` (dossier `plots/`, créé au
  premier lancement, non versionné).
- **Console** : loss et accuracy par epoch.
- **Modèles** : voir ci-dessus (`models/`).
