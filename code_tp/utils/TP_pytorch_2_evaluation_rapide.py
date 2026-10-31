"""TP 2 - Evaluation rapide du modele chat / pas chat.

Charge le modele entraine (models/<MODEL_NAME>) et affiche, pour chaque
image de data/test_images, la classe predite et le score (softmax).
Pas de calcul d'accuracy : on regarde les predictions image par image.

Prerequis : avoir lance l'entrainement (TP_pytorch_2_training*.py).
Usage : python TP_pytorch_2_evaluation_rapide.py
"""

import glob
import os
import sys
import torch
from torchvision.models import ResNet18_Weights
from PIL import Image

# parametre : modele a tester (dans models/) : cat_model_init.pth ou cat_model_best.pth
MODEL_NAME = 'cat_model_init.pth'

def list_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, '**', f'*{ext}'), recursive=True))
    return files


# Construct relative paths
base_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
eval_dataset_dir = os.path.join(parent_dir, 'data/test_images')
models_dir = os.path.join(parent_dir, 'models')
save_model_path = os.path.join(models_dir, MODEL_NAME)


# main du TP
if __name__ == '__main__':

    # verification : le modele a tester doit exister
    if not os.path.isfile(save_model_path):
        print(f"ERREUR : modele introuvable : {save_model_path}")
        if os.path.isdir(models_dir):
            print(f"Modeles disponibles dans {models_dir} : {sorted(os.listdir(models_dir))}")
        else:
            print("Le dossier models/ n'existe pas : lance d'abord l'entrainement (TP_pytorch_2_training*.py).")
        print("Verifie MODEL_NAME en haut du fichier.")
        sys.exit(1)

    # on charge le models qu on vient d entrainer
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    mon_model_trained = torch.load(save_model_path,  weights_only=False)
    mon_model_trained.eval()
    categories = mon_model_trained.categories
    print(mon_model_trained.categories)

    files = list_image_files(eval_dataset_dir)
    for image in files:
        mon_image = Image.open(image)
        image_name = image.split('/')[-1]


        image_preprocessed = preprocess(mon_image)
        batch = image_preprocessed.unsqueeze(0)

        prediction_non_formatee = mon_model_trained(batch)

        prediction = prediction_non_formatee.squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score_prediction = prediction[class_id].item()
        predicted_class = categories[class_id]

        print(f"{image_name:<25} | {predicted_class:<20} | {score_prediction:<10.3f}")
        print("=" * 60)
