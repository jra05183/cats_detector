# TP 3 — Entraînement outillé : suivi, scheduler, arrêt anticipé

## But du TP

Même problème que le TP 2 (chat / pas chat), mais on apprend à **observer** l'entraînement et à le
piloter :

- découpage train / validation équilibré et augmentation d'images (sur le train seulement) ;
- scheduler de learning rate (`ReduceLROnPlateau`) ;
- arrêt anticipé (early stopping) avec retour aux meilleurs poids ;
- dégel optionnel de `layer4` (fine-tuning partiel) ;
- suivi des courbes dans TensorBoard ;
- test final sur des images jamais utilisées pour choisir le modèle.

## Fichier à utiliser

| Fichier | Rôle |
|---------|------|
| `TP_pytorch_3_training.py` | Squelette à compléter : **5 `TODO`** (dégel de `layer4`, groupes de l'optimiseur, scheduler, copie des meilleurs poids, condition d'arrêt) |

Lancement, depuis la racine du projet (avec le venv du projet) :

```bash
python code_tp/TP_3/TP_pytorch_3_training.py
```

Options pour comparer plusieurs essais sans modifier le fichier :

```bash
python code_tp/TP_3/TP_pytorch_3_training.py --run-name B --no-augmentation
python code_tp/TP_3/TP_pytorch_3_training.py --run-name C --unfreeze-layer4
python code_tp/TP_3/TP_pytorch_3_training.py --run-name D --unfreeze-layer4 --no-augmentation
```

Les réglages (learning rates, patience, nombre d'epochs…) sont dans le dictionnaire `cfg` en haut
du fichier.

## Images utilisées

- `images/cat/` et `images/no_cat/` : entraînement et validation (découpage 80 % / 20 %, la proportion
  de chaque classe est conservée).
- `data/test_images/` : **test final**, lu une seule fois à la fin.

## Où est sauvegardé le modèle ?

`models/cat_model_tp3_<run_name>.pth` : le modèle des **meilleurs poids** (pas ceux de la dernière
epoch). Un fichier par run, rien n'est écrasé. Pour l'évaluer, mettre ce nom dans `MODEL_NAME` des
scripts de [`../utils/`](../utils/README.md).

## Où sont les résultats ?

Dans `runs/<run_name>/` (dossier créé au premier lancement, non versionné), **pas dans `plots/`** :

| Fichier | Contenu |
|---------|---------|
| événements TensorBoard | courbes, learning rate, images augmentées, histogrammes des poids |
| `metrics.csv` | toutes les métriques par epoch |
| `curves.png` | loss, accuracy et learning rate |
| `confusion_validation.png`, `confusion_test.png` | matrices de confusion |
| `results.json` | résumé du run (meilleur epoch, accuracy de test, images mal classées) |

Pour visualiser et comparer les runs :

```bash
tensorboard --logdir runs
```

Si TensorBoard n'est pas accessible (serveur distant sans redirection de port), ouvrir
`curves.png` et `metrics.csv`.
