"""TP 2 - Evaluation complete du modele chat / pas chat.

Charge le modele entraine (models/<MODEL_NAME>), le fait predire sur
data/test_images (dossiers cat/ et no_cat/ = verite terrain), puis affiche
l'accuracy et la matrice de confusion.

Prerequis : avoir lance l'entrainement (TP_pytorch_2_training*.py).
Usage : python TP_pytorch_2_evaluation_full.py
"""

from torchvision.models import ResNet18_Weights

import os
import sys

import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from torchvision.datasets import ImageFolder


# Get the directory of the current script
base_dir = os.path.dirname(__file__)


#------------------------------
# parametres utilisés dans ce fichier
MODEL_NAME = 'cat_model_init.pth'  # modele a tester (dans models/) : cat_model_init.pth ou cat_model_best.pth
batch_size = 4 #taille du batch pour l'entrainement

# Construct relative paths
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
eval_dataset_dir = os.path.join(parent_dir, 'data/test_images')
models_dir = os.path.join(parent_dir, 'models')
save_model_path = os.path.join(models_dir, MODEL_NAME)
plots_dir = os.path.join(parent_dir, 'plots')  # les figures sont sauvegardees ici (en .png)

device = torch.device('cpu')



def get_dataset(preprocess):
    # transforme les deux repos avec les images en dataset labelisé
    # 2 - creation d'un dataset chat / pas chat a partir du repertoire images
    # preprocess est passe en parametre (et non lu comme variable globale) :
    # la fonction depend uniquement de ses arguments, on peut la reutiliser ailleurs
    dataset = ImageFolder(root=eval_dataset_dir, transform=preprocess)
    # shuffle=False : en evaluation on ne met rien a jour, l'ordre des images n'a pas
    # d'importance (le melange ne sert que pendant l'entrainement) ; sans melange les
    # resultats sont toujours dans le meme ordre, donc plus faciles a comparer
    evaluation_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return evaluation_dataset



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

    # on a besoin d'un models
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    mon_model_trained = torch.load(save_model_path, weights_only=False)
    mon_model_trained.to(device)

    # on a besoin d'un dataset de test (data/test_images) : pas de split ici,
    # ces images n'ont pas servi a l'entrainement et leur dossier (cat / no_cat) donne le label
    evaluation_dataloader = get_dataset(preprocess)
    nb_prediction_ok = 0
    num_eval_samples = 0

    # Evaluate
    mon_model_trained.eval()
    all_preds = []
    all_labels = []
    all_scores = []  # confiance (softmax) du modele dans sa prediction, pour chaque image
    with torch.no_grad():
        for xtest, ytest in evaluation_dataloader:
            xtest = xtest.to(device)
            ytest = ytest.to(device)
            test_prob = mon_model_trained(xtest)
            test_pred = torch.max(test_prob, 1).indices
            nb_prediction_ok += int(torch.sum(test_pred == ytest))
            num_eval_samples += ytest.size(0)
            all_preds.extend(test_pred.cpu().numpy())
            all_labels.extend(ytest.cpu().numpy())
            all_scores.extend(test_prob.softmax(dim=1).max(dim=1).values.cpu().numpy())

    test_accuracy = (nb_prediction_ok / num_eval_samples) * 100
    print(f"Test accuracy: {test_accuracy:.4f}")


    # noms des classes dans l'ordre des indices : {0: 'cat', 1: 'no_cat'} => ['cat', 'no_cat']
    # (le modele les a gardes a l'entrainement dans l'attribut categories)
    class_names = [mon_model_trained.categories[i] for i in range(len(mon_model_trained.categories))]

    # Calculate the confusion matrix
    # labels=... force une matrice 2 x 2 meme si une classe n'apparait pas dans les predictions
    # lignes = classe reelle, colonnes = classe predite
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    print(conf_matrix)

    # Rapport par classe : l'accuracy seule cache les differences entre les classes
    # precision = parmi les images annoncees "cat", combien sont vraiment des chats
    # recall (rappel) = parmi les vrais chats, combien le modele en a retrouve
    # f1-score = compromis entre precision et recall ; support = nombre d'images reelles de la classe
    print("\nRapport par classe :")
    print(classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                target_names=class_names, digits=3, zero_division=0))

    # Images mal classees : on veut savoir lesquelles se sont trompees, pas seulement combien
    # Comme shuffle=False, la i-eme prediction correspond a la i-eme image de dataset.samples
    # (liste de (chemin du fichier, label)) : c'est ce qui permet de retrouver le nom du fichier
    samples = evaluation_dataloader.dataset.samples
    errors = [i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]
    print(f"Images mal classees : {len(errors)} / {len(all_labels)}")
    for i in errors:
        image_name = os.path.basename(samples[i][0])
        print(f"{image_name:<25} | reel : {class_names[all_labels[i]]:<8} | "
              f"predit : {class_names[all_preds[i]]:<8} | score : {all_scores[i]:.3f}")
    print()

    # Plot the confusion matrix
    # xticklabels / yticklabels : on affiche cat / no_cat au lieu de 0 / 1 pour lire la matrice facilement
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # sauvegarde de la figure en .png : sur un serveur sans ecran (ou en SSH) plt.show() ne
    # peut rien afficher, alors qu'on peut toujours ouvrir le .png dans l'editeur
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"confusion_matrix_{os.path.splitext(MODEL_NAME)[0]}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Figure sauvegardee : {plot_path}")

    # affichage a l'ecran seulement si matplotlib a une fenetre disponible (pas le backend Agg)
    if 'agg' not in plt.get_backend().lower():
        plt.show()






