import glob
import os

from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image


def list_files(directory):
    return glob.glob(directory + '/**/*', recursive=True)



# utilisation d'un models pre-entraine pour la classification d'image nom = ResNet101
# test avec une image
# on a besoin a la fois de l'archi du models et des poids

# get pre-trained models weights
weights = ResNet101_Weights.DEFAULT

# le models a des infos dans ses meta data
# categories donne les classes que le models sait classer.
# On a 1000 classes chacune a un id qui donne sa place dans le tensor de sortie
print('nombre de classes que le models sait classifier = {}'.format(len(weights.meta["categories"])))
print('nom des classes d\'id entre 5 et 12 : {}'.format(weights.meta["categories"][5:12]))

# le resnet attend que les images en entree soient des tensor normalised avec comme taille 3 x 234 x 234 => C x H x W
# C = channel : 3 car image couleur RGB
# H = hauteur et W = largeur : 224 x 224
# la class Resnet101 propose directement le pretraitement necessaire pour pouvoir appeler le models
preprocess = weights.transforms()

# instanciation du models avec les poids et passage en models eval => notre but est l'utilisation pas le training pour le moment
mon_model = resnet101(weights=weights)

# le models est au format pytorch et est un graph de calcul
#print('models = {}'.format(mon_model))

# on met le models en mode eval pour eviter que les poids ne soient modifies
mon_model.eval()

# recuperation de l'image de test que l'on veut faire classer par le models
# get all file from a specific repository
base_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
directory = os.path.join(parent_dir, './data/test_images_2')
files = list_files(directory=directory)
for image in files:
    mon_image = Image.open(image)
    image_name = image.split('/')[-1]

    # application du preprocessing a l'image de test
    image_preprocessed = preprocess(mon_image)
    # check du shape de sortie
    # print('shape de l  image apres le preprocess {}'.format({image_preprocessed.shape}))

    # formatage de l'input pour quelle passe bien dans le models
    batch = image_preprocessed.unsqueeze(0)
    # print('shape de l input du models {}'.format({batch.shape}))

    # appel du forward du models : resultat = prediction de la classe format = brut
    prediction_non_formatee = mon_model(batch)

    # formatage de la prediction => on enleve la dimension 1 qui nous sert a rien
    # en sortie on a un tensor de la meme taille que le nombre de classe que le models prend en compte = 1000
    # pour chaque element la valeur = pourcentage de change que la photo soit de la classe
    # exemple l'element avec l'id 282 = class tiger cat
    # la classe la plus probable est celle qui a la proba la plus grande => on utilise sofmax pour avec l'id de la classe qui a la plus grande proba
    prediction = prediction_non_formatee.squeeze(0).softmax(0)
    # print('shape de la sortie du models (prediction) = {}'.format(prediction.shape))

    # recuperation du class_id de la prediction
    class_id = prediction.argmax().item()
    # print('class_id de l element du tensor de sortie qui a la plus grande proba = {}'.format(class_id))
    # recuperation de la valeur de proba
    score_prediction = prediction[class_id].item()
    # print('proba de la classe qui a la plus forte proba = {}'.format(score_prediction))

    print('Prediction : l image {} est un {} avec la proba {}'.format(image_name, weights.meta["categories"][class_id], score_prediction))



