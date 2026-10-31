"""TP 1 - Inference avec un ResNet-18 pre-entraine (ImageNet, 1000 classes).

Squelette a completer (voir les TODO) : charge le modele, applique le
preprocessing aux images de data/test_images, fait le forward et doit
afficher la classe ImageNet la plus probable pour chaque image.

Usage : python TP_pytorch_1_init.py
"""
import glob
import os

import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


def list_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, '**', f'*{ext}'), recursive=True))
    return files


def main():
    # utilisation d'un models pre-entraine pour la classification d'image nom = ResNet18
    # test avec une image
    # on a besoin a la fois de l'archi du models et des poids

    # get pre-trained models weights
    weights = ResNet18_Weights.DEFAULT

    # le models a des infos dans ses meta data
    # categories donne les classes que le models sait classer.
    # On a 1000 classes chacune a un id qui donne sa place dans le tensor de sortie
    print('nombre de classes que le models sait classifier = {}'.format(len(weights.meta["categories"])))
    print('nom des classes d\'id entre 5 et 12 : {}'.format(weights.meta["categories"][5:12]))

    # le resnet attend que les images en entree soient des tensor normalised avec comme taille 3 x 224 x 224 => C x H x W
    # C = channel : 3 car image couleur RGB
    # H = hauteur et W = largeur : 224 x 224
    # la class ResNet18 propose directement le pretraitement necessaire pour pouvoir appeler le models
    preprocess = weights.transforms()

    # instanciation du models avec les poids et passage en models eval => notre but est l'utilisation pas le training pour le moment
    mon_model = resnet18(weights=weights)

    # le models est au format pytorch et est un graph de calcul
    # print('models = {}'.format(mon_model))

    # on met le models en mode eval pour eviter que les poids ne soient modifies
    mon_model.eval()

    # recuperation de l'image de test que l'on veut faire classer par le models
    # get all file from a specific repository
    base_dir = os.path.dirname(__file__)
    directory = os.path.join(base_dir, '../../data/test_images')
    files = list_image_files(directory)
    for image in files:
        mon_image = Image.open(image).convert('RGB')
        image_name = image.split('/')[-1]

        # application du preprocessing a l'image de test
        image_preprocessed = preprocess(mon_image)
        # check du shape de sortie
        print(f'shape de l image apres le preprocess {image_preprocessed.shape}')

        # formatage de l'input pour quelle passe bien dans le models
        batch = image_preprocessed.unsqueeze(0)
        print(f'shape de l input du models {batch.shape}')

        # appel du forward du models : resultat = prediction de la classe format = brut
        # inference seulement : pas besoin de calculer les gradients (economie memoire et temps)
        with torch.no_grad():
            prediction_non_formatee = mon_model(batch)

        # TODO
        # faire un print du models => regarder comment pytorch représente le models sous forme de graph de calcul

        # TODO
        # analyser le shape de la sortie du models

        # TODO
        # trouver quelle est la prédiction du models => la classe la plus probable

        # TODO
        # trouver le nom de la classe prédite (dans les metadata de weights il y a les noms des classes par ordre)


if __name__ == '__main__':
    main()
