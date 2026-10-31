# cats_detector

Classification binaire **chat / pas chat** par transfer learning sur **ResNet-18** (PyTorch).

## Prérequis

- **Python 3.10** (recommandé)
- Environnement virtuel conseillé

## Installation

```bash
cd cats_detector
python3.10 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Si l’installation de `torch` / `torchvision` échoue, suivre les instructions [PyTorch](https://pytorch.org/get-started/locally/) (CPU ou GPU), puis relancer `pip install -r requirements.txt` pour le reste.

## Données

| Dossier | Usage |
|---------|--------|
| `images/cat`, `images/no_cat` | Entraînement (TP 2) |
| `data/test_images/` | Images de test (TP 1 et évaluation) |

Les modèles entraînés sont écrits dans `models/` (créé au premier entraînement).

## Déroulé des scripts

### TP 1 — Inférence ImageNet (1000 classes)

| Fichier | Rôle |
|---------|------|
| `code_tp/TP_1/TP_pytorch_1_init.py` | Squelette avec TODO |
| `code_complet/TP_1/TP_pytorch_1_full.py` | Corrigé |

```bash
cd code_tp/TP_1
python TP_pytorch_1_init.py

cd ../../code_complet/TP_1
python TP_pytorch_1_full.py
```

### TP 2 — Fine-tuning binaire, évaluation

| Fichier | Rôle |
|---------|------|
| `code_tp/TP_2/TP_pytorch_2_training.py` | Entraînement (TODO) |
| `code_complet/TP_2/TP_pytorch_2_training_full.py` | Corrigé entraînement |
| `code_tp/utils/TP_pytorch_2_evaluation_rapide.py` | Prédiction image par image |
| `code_tp/utils/TP_pytorch_2_evaluation_full.py` | Accuracy + matrice de confusion |

```bash
cd code_complet/TP_2
python TP_pytorch_2_training_full.py

cd ../../code_tp/utils
python TP_pytorch_2_evaluation_rapide.py
python TP_pytorch_2_evaluation_full.py
```

Après un changement d’architecture (ex. passage à ResNet-18), **réentraîner** avant d’évaluer : les fichiers `models/cat_model_*.pth` ne sont pas interchangeables entre architectures.

### TP 3 — Entraînement outillé (suivi, scheduler, arrêt anticipé)

| Fichier | Rôle |
|---------|------|
| `code_tp/TP_3/TP_pytorch_3_training.py` | Entraînement (5 TODO) |
| `code_complet/TP_3/TP_pytorch_3_training_full.py` | Corrigé |

Même problème que le TP 2, avec : découpage train / validation stratifié, augmentation d’images (train seulement), scheduler de learning rate, arrêt anticipé, dégel optionnel de `layer4`, suivi TensorBoard et test final unique.

```bash
cd code_complet/TP_3
python TP_pytorch_3_training_full.py                                   # A : backbone gelé + augmentation
python TP_pytorch_3_training_full.py --run-name B --no-augmentation    # B : gelé, sans augmentation
python TP_pytorch_3_training_full.py --run-name C --unfreeze-layer4    # C : layer4 dégelé + augmentation
python TP_pytorch_3_training_full.py --run-name D --unfreeze-layer4 --no-augmentation

tensorboard --logdir ../../runs    # compare les runs sur les mêmes graphiques
```

Chaque run écrit dans `runs/<run_name>/` (`metrics.csv`, `curves.png`, matrices de confusion, `results.json`, événements TensorBoard) et exporte `models/cat_model_tp3_<run_name>.pth`. Pour l’évaluer, mettre ce nom dans `MODEL_NAME` des scripts du TP 2. Un run existant n’est jamais écrasé.

Sans TensorBoard (serveur distant sans redirection de port), ouvrir `curves.png` et `metrics.csv`.

## Support de cours

- Le support de cours est distribué séparément (il n'est pas dans le dépôt).
- Énoncés : distribués séparément (les `.docx` ne sont plus dans `code_tp/`).
