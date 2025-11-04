import glob
import os
import torch
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image

def list_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, '**', f'*{ext}'), recursive=True))
    return files


# Construct relative paths
base_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
eval_dataset_dir = os.path.join(parent_dir, 'data/test_images')
save_model_path = os.path.join(parent_dir, 'models/cat_model_init.pth')


# main du TP
if __name__ == '__main__':

    # on charge le models qu on vient d entrainer
    weights = ResNet101_Weights.DEFAULT
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
