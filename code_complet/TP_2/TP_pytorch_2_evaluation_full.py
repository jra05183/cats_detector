
from torchvision.models import resnet101, ResNet101_Weights

import os

import torch

from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix

from torchvision.datasets import ImageFolder


# Get the directory of the current script
base_dir = os.path.dirname(__file__)


#------------------------------
# parametres utilisés dans ce fichier
batch_size = 4 #taille du batch pour l'entrainement

# Construct relative paths
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
eval_dataset_dir = os.path.join(parent_dir, 'data/eval_inference')
save_model_path = os.path.join(parent_dir, 'models/cat_model_init.pth')

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_dataset():
    # transforme les deux repos avec les images en dataset labelisé
    # 2 - creation d'un dataset chat / pas chat a partir du repertoire images
    dataset = ImageFolder(root=eval_dataset_dir, transform=preprocess)
    evaluation_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return evaluation_dataset



# main du TP
if __name__ == '__main__':

    # on a besoin d'un models
    weights = ResNet101_Weights.DEFAULT
    preprocess = weights.transforms()
    mon_model_trained = torch.load(save_model_path,  weights_only=False)


    # on a besoin d'un dataset et qu'il soit divisé en 2 : training et validation
    evaluation_dataloader = get_dataset()
    len_val = len(evaluation_dataloader)
    # initialisation des variables pour les metriques accuracy du training et du test
    nb_prediction_ko_current_epoch_validation = 0

    # Evaluate
    mon_model_trained.eval()
    validation_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xtest, ytest in evaluation_dataloader:
            xtest = xtest.to(device)
            ytest = ytest.to(device)
            test_prob = mon_model_trained(xtest)
            test_pred = torch.max(test_prob, 1).indices
            nb_prediction_ko_current_epoch_validation += int(torch.sum(test_pred == ytest))
            all_preds.extend(test_pred.cpu().numpy())
            all_labels.extend(ytest.cpu().numpy())
            print(f"Validation accuracy: {nb_prediction_ko_current_epoch_validation}")

    # accuracy pour l'epoque en cours = nombre de prediction correcte / nombre total de prediction = taille du dataset de validation
    nombre_total_image_validation_set = len_val*batch_size
    validation_accuracy = nb_prediction_ko_current_epoch_validation / nombre_total_image_validation_set


    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(conf_matrix)

    # plot accuracy per epoch
    # accurary for training and validation

    # Plot the confusion matrix
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()






