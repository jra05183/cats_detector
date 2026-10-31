# Utils — Évaluation des modèles entraînés

## But

Tester un modèle chat / pas chat déjà entraîné (TP 2 ou TP 3) sur les images de test, sans
réentraîner.

## Fichiers à utiliser

| Fichier | Rôle |
|---------|------|
| `TP_pytorch_2_evaluation_rapide.py` | Prédiction **image par image** : classe prédite et score |
| `TP_pytorch_2_evaluation_full.py` | **Accuracy**, rapport par classe, matrice de confusion et liste des images mal classées |

Lancement, depuis la racine du projet (avec le venv du projet) :

```bash
python code_tp/utils/TP_pytorch_2_evaluation_rapide.py
python code_tp/utils/TP_pytorch_2_evaluation_full.py
```

**Choisir le modèle à tester** : modifier `MODEL_NAME` en haut de chaque fichier. Valeurs possibles :
`cat_model_init.pth` ou `cat_model_best.pth` (TP 2), `cat_model_tp3_<run_name>.pth` (TP 3). Si le
fichier est absent, le script affiche une erreur et la liste des modèles disponibles.

## Images utilisées

`data/test_images/` (sous-dossiers `cat/` et `no_cat/` = vérité terrain).

## Où est le modèle lu ?

Dans `models/` (créé par l'entraînement). Il faut donc avoir lancé le TP 2 ou le TP 3 avant.

## Où sont les résultats ?

- **Évaluation rapide** : uniquement dans le terminal.
- **Évaluation complète** : terminal (accuracy, rapport, images mal classées) et figure de la matrice de
  confusion dans `plots/confusion_matrix_<MODEL_NAME sans .pth>.png` (dossier `plots/`, créé au premier
  lancement, non versionné).
