"""TP 3 (corrige) - Entrainement "outille" de ResNet-18 : chat / pas chat.

Meme probleme que le TP 2, mais on apprend a OBSERVER l'entrainement :
  - decoupage train / validation stratifie (autant de chats des deux cotes),
  - augmentation d'images sur le train seulement (option),
  - scheduler de learning rate (ReduceLROnPlateau),
  - arret anticipe (early stopping) avec restauration des meilleurs poids,
  - degel optionnel de layer4 (fine-tuning partiel, option),
  - suivi dans TensorBoard (+ metrics.csv et curves.png en secours),
  - test final sur data/test_images, utilise UNE seule fois.

Chaque lancement ecrit dans runs/<run_name>/ et models/ (rien n'est ecrase).

Usage :
    python TP_pytorch_3_training_full.py
    python TP_pytorch_3_training_full.py --run-name C --unfreeze-layer4
    python TP_pytorch_3_training_full.py --run-name D --unfreeze-layer4 --no-augmentation
    tensorboard --logdir runs
"""
import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.utils import make_grid

# ------------------------------
# parametres du TP : tout se regle ici (ou avec les options de la ligne de commande)
cfg = {
    'run_name': 'gele_aug',        # nom du run : dossier runs/<run_name>/ et modele models/cat_model_tp3_<run_name>.pth
    'unfreeze_layer4': False,      # True : layer4 apprend aussi (fine-tuning partiel)
    'unfreeze_at_epoch': 3,        # epoch a partir de laquelle layer4 est degele (phase 2)
    'use_augmentation': True,      # augmentation d'images sur le train (pas sur la validation)
    'batch_size': 8,               # taille du batch
    'max_epochs': 30,              # plafond : l'arret anticipe coupe en general avant
    'ratio': 0.2,                  # part des images mise en validation
    'seed': 42,                    # graine aleatoire (split, init, augmentation)
    'lr_head': 1e-3,               # learning rate de la derniere couche (fc)
    'lr_layer4': 1e-4,             # learning rate de layer4 : 10 fois plus petit pour ne pas casser ce qu'il sait deja
    'weight_decay': 1e-2,          # regularisation de AdamW
    'grad_clip': 5.0,              # on coupe les gradients dont la norme depasse cette valeur
    'scheduler_patience': 3,       # nb d'epochs sans progres avant de diviser le learning rate
    'scheduler_factor': 0.5,       # le learning rate est multiplie par ce facteur
    'early_stop_patience': 6,      # nb d'epochs sans progres avant d'arreter (doit etre > scheduler_patience)
    'early_stop_min_delta': 1e-3,  # progres minimal de la loss de validation pour compter comme une amelioration
    'hist_every': 5,               # tous les N epochs : histogrammes des poids / gradients dans TensorBoard
}

# ------------------------------
# chemins relatifs au projet
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
data_dir = os.path.join(parent_dir, 'images')
data_test_dir = os.path.join(parent_dir, 'data', 'test_images')
models_dir = os.path.join(parent_dir, 'models')
runs_dir = os.path.join(parent_dir, 'runs')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parametres dont on suit les histogrammes dans TensorBoard (la derniere couche + une conv de layer4)
HIST_PARAMS = ('fc.weight', 'layer4.1.conv2.weight')


def parse_args():
    # options facultatives pour lancer plusieurs essais sans modifier le fichier
    parser = argparse.ArgumentParser(description="TP3 : entrainement outille chat / pas chat")
    parser.add_argument('--run-name', default=cfg['run_name'])
    parser.add_argument('--unfreeze-layer4', action=argparse.BooleanOptionalAction,
                        default=cfg['unfreeze_layer4'])
    parser.add_argument('--augmentation', action=argparse.BooleanOptionalAction,
                        default=cfg['use_augmentation'])
    return parser.parse_args()


def set_seed(seed):
    # meme graine partout : python, numpy et torch (split, init de fc, augmentation, shuffle)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def unique_run_name(name):
    # on n'ecrase jamais un run existant : si le dossier existe deja, on ajoute l'heure
    if os.path.exists(os.path.join(runs_dir, name)):
        name = f"{name}_{datetime.now().strftime('%H%M%S')}"
        print(f"Le run existe deja, nouveau nom : {name}")
    return name


# ------------------------------
# 1 - modele

def instanciate_model():
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    mon_model = resnet18(weights=weights)
    # on gele tout le reseau pre-entraine
    for param in mon_model.parameters():
        param.requires_grad = False
    # nouvelle derniere couche : 2 sorties (chat / pas chat), elle est entrainable
    mon_model.fc = nn.Linear(in_features=mon_model.fc.in_features, out_features=2)
    nn.init.normal_(mon_model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(mon_model.fc.bias)
    return mon_model.to(device), preprocess


def set_layer4_trainable(model, trainable):
    # degel (ou gel) du dernier bloc du ResNet : c'est le fine-tuning partiel
    for param in model.layer4.parameters():
        param.requires_grad = trainable


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_train_mode(model):
    # model.train() remet aussi le BatchNorm en mode "entrainement" : ses statistiques internes
    # seraient recalculees sur des batchs minuscules (bruit). On les garde donc figees.
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


# ------------------------------
# 2 - donnees

def stratified_split(targets, ratio, seed):
    # decoupe train / validation en gardant la meme proportion de chaque classe des deux cotes
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for cls in sorted(set(targets)):
        idx = [i for i, t in enumerate(targets) if t == cls]
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * ratio))
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]
    return sorted(train_idx), sorted(val_idx)


def get_data(preprocess):
    # deux ImageFolder sur le MEME repertoire : un avec augmentation (train), un sans (validation)
    if cfg['use_augmentation']:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(preprocess.crop_size[0], scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(preprocess.mean, preprocess.std),
        ])
    else:
        train_transform = preprocess
    train_full = ImageFolder(root=data_dir, transform=train_transform)
    eval_full = ImageFolder(root=data_dir, transform=preprocess)

    idx_to_class = {v: k for k, v in train_full.class_to_idx.items()}
    print("Index to class mapping:", idx_to_class)

    train_idx, val_idx = stratified_split(train_full.targets, cfg['ratio'], cfg['seed'])
    train_set = Subset(train_full, train_idx)   # images augmentees
    val_set = Subset(eval_full, val_idx)        # memes indices, mais sans augmentation
    val_files = [eval_full.samples[i][0] for i in val_idx]
    print(f"Train : {len(train_idx)} images, validation : {len(val_idx)} images")

    test_set = ImageFolder(root=data_test_dir, transform=preprocess)
    assert test_set.class_to_idx == train_full.class_to_idx, "classes differentes entre train et test"
    test_files = [path for path, _ in test_set.samples]

    generator = torch.Generator().manual_seed(cfg['seed'])
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False)
    return train_loader, val_loader, test_loader, idx_to_class, val_files, test_files


# ------------------------------
# 3 - outils d'entrainement

class EarlyStopping:
    """Arrete l'entrainement quand la loss de validation ne progresse plus."""

    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.best_state = None   # copie des meilleurs poids
        self.counter = 0         # nb d'epochs consecutives sans progres
        self.should_stop = False

    def update(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            # progres : on retient cet epoch et on copie les poids
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def get_optimizer(model):
    # deux groupes de parametres : la derniere couche apprend vite, layer4 (si degele) apprend lentement.
    # Tant que layer4 est gele, ses parametres n'ont pas de gradient : l'optimiseur les ignore.
    return AdamW([
        {'params': model.fc.parameters(), 'lr': cfg['lr_head']},
        {'params': model.layer4.parameters(), 'lr': cfg['lr_layer4']},
    ], weight_decay=cfg['weight_decay'])


def run_epoch(model, loader, loss_fn, optimiser=None):
    """Un passage sur `loader`. Avec un optimiseur : entrainement, sinon : evaluation."""
    training = optimiser is not None
    if training:
        set_train_mode(model)
    else:
        model.eval()
    total_loss, correct, n_samples = 0.0, 0, 0
    grad_norms, all_probs, all_labels = [], [], []
    with torch.enable_grad() if training else torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            if training:
                optimiser.zero_grad()
                loss.backward()
                params = [p for p in model.parameters() if p.requires_grad]
                # clip_grad_norm_ coupe les gradients trop grands et renvoie la norme AVANT coupe
                grad_norms.append(float(nn.utils.clip_grad_norm_(params, cfg['grad_clip'])))
                optimiser.step()
            total_loss += loss.item() * y.size(0)   # moyenne ponderee (le dernier batch est plus petit)
            correct += int((output.argmax(dim=1) == y).sum())
            n_samples += y.size(0)
            all_probs.append(torch.softmax(output.detach(), dim=1).cpu())
            all_labels.append(y.cpu())
    return {
        'loss': total_loss / n_samples,
        'acc': correct / n_samples,
        'grad_norm': float(np.mean(grad_norms)) if grad_norms else 0.0,
        'probs': torch.cat(all_probs).numpy(),
        'labels': torch.cat(all_labels).numpy(),
    }


# ------------------------------
# 4 - suivi : TensorBoard + metrics.csv

class RunLogger:
    """Ecrit les metriques dans TensorBoard (si installe) et dans metrics.csv."""

    def __init__(self, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.csv_path = os.path.join(run_dir, 'metrics.csv')
        self.csv_started = False
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=run_dir)
        except ImportError:
            self.writer = None
            print("TensorBoard n'est pas installe : seuls metrics.csv et les .png sont ecrits.")

    def scalars(self, epoch, row):
        # une ligne dans le csv + un point par courbe dans TensorBoard
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not self.csv_started:
                writer.writerow(['epoch'] + list(row.keys()))
                self.csv_started = True
            writer.writerow([epoch] + [f"{v:.6g}" for v in row.values()])
        if self.writer:
            for tag, value in row.items():
                self.writer.add_scalar(tag, value, epoch)

    def histograms(self, model, epoch):
        if not self.writer:
            return
        for name, param in model.named_parameters():
            if name in HIST_PARAMS and param.requires_grad:
                self.writer.add_histogram(f'weights/{name}', param.detach().cpu(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'grads/{name}', param.grad.detach().cpu(), epoch)

    def image(self, tag, tensor):
        if self.writer:
            self.writer.add_image(tag, tensor)

    def figure(self, tag, fig, filename):
        fig.savefig(os.path.join(self.run_dir, filename), bbox_inches='tight')
        if self.writer:
            self.writer.add_figure(tag, fig)
        plt.close(fig)

    def text(self, tag, text):
        if self.writer:
            # 4 espaces en debut de ligne : TensorBoard affiche le texte tel quel
            self.writer.add_text(tag, '\n'.join('    ' + line for line in text.splitlines()))

    def close(self):
        if self.writer:
            self.writer.close()


def log_augmented_batch(logger, loader, preprocess):
    # on regarde ce que voit vraiment le reseau : un batch apres augmentation
    images, _ = next(iter(loader))
    mean = torch.tensor(preprocess.mean).view(3, 1, 1)
    std = torch.tensor(preprocess.std).view(3, 1, 1)
    grid = make_grid((images[:8] * std + mean).clamp(0, 1), nrow=4)
    logger.image('donnees/batch_train_apres_augmentation', grid)


def weight_norms(model):
    norm_fc = float(model.fc.weight.detach().norm())
    norm_l4 = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.layer4.parameters())))
    return norm_fc, norm_l4


def confusion_figure(labels, preds, class_names, title):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predit')
    ax.set_ylabel('Reel')
    ax.set_title(title)
    return fig


def misclassified_figure(files, labels, preds, probs, class_names, max_images=8):
    wrong = [i for i in range(len(files)) if labels[i] != preds[i]][:max_images]
    if not wrong:
        return None
    cols = min(4, len(wrong))
    rows = math.ceil(len(wrong) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(Image.open(files[i]).convert('RGB'))
        ax.set_title(f"vrai : {class_names[labels[i]]}\npredit : {class_names[preds[i]]} "
                     f"({probs[i][preds[i]]:.2f})", fontsize=8)
    fig.tight_layout()
    return fig


def save_curves(history, run_dir):
    # courbes de secours (si TensorBoard n'est pas disponible) : loss, accuracy, learning rate
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (key, title) in zip(axes[:2], [('loss', 'Loss'), ('acc', 'Accuracy')]):
        ax.plot(epochs, [h[f'{key}/train'] for h in history], label='Train')
        ax.plot(epochs, [h[f'{key}/val'] for h in history], label='Validation')
        ax.set_title(title + ' par epoch')
        ax.set_xlabel('Epoch')
        ax.legend()
    axes[2].plot(epochs, [h['lr/head'] for h in history], label='fc')
    axes[2].plot(epochs, [h['lr/layer4'] for h in history], label='layer4')
    axes[2].set_yscale('log')
    axes[2].set_title('Learning rate')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    fig.savefig(os.path.join(run_dir, 'curves.png'), bbox_inches='tight')
    plt.close(fig)


def safe_auc(labels, probs):
    # AUC-ROC de la classe d'indice 1 (calculable seulement si les deux classes sont presentes)
    try:
        return float(roc_auc_score(labels, probs[:, 1]))
    except ValueError:
        return None


# ------------------------------
# main du TP
def main():
    args = parse_args()
    cfg.update(run_name=args.run_name, unfreeze_layer4=args.unfreeze_layer4,
               use_augmentation=args.augmentation)
    assert cfg['early_stop_patience'] > cfg['scheduler_patience'], \
        "early_stop_patience doit etre > scheduler_patience (sinon on arrete avant que le learning rate baisse)"
    set_seed(cfg['seed'])

    run_name = unique_run_name(cfg['run_name'])
    run_dir = os.path.join(runs_dir, run_name)
    logger = RunLogger(run_dir)
    print(f"Run : {run_name}   (dossier : {run_dir})")
    print(f"Config : {json.dumps({k: cfg[k] for k in ('unfreeze_layer4', 'use_augmentation', 'seed')})}")
    logger.text('config', json.dumps(cfg, indent=2))

    # modele, donnees, optimiseur, loss, scheduler, arret anticipe
    mon_model, preprocess = instanciate_model()
    train_loader, val_loader, test_loader, idx_to_class, val_files, test_files = get_data(preprocess)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    optimiser = get_optimizer(mon_model)
    loss_fn = CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=cfg['scheduler_factor'],
                                  patience=cfg['scheduler_patience'])
    stopper = EarlyStopping(cfg['early_stop_patience'], cfg['early_stop_min_delta'])
    print(f"Parametres entrainables au depart : {count_trainable(mon_model)}")
    log_augmented_batch(logger, train_loader, preprocess)

    # ------------------------------
    # boucle d'entrainement
    history = []
    start_total = time()
    for epoch in range(cfg['max_epochs']):
        if cfg['unfreeze_layer4'] and epoch == cfg['unfreeze_at_epoch']:
            set_layer4_trainable(mon_model, True)
            print(f"--> epoch {epoch} : layer4 degele, parametres entrainables = {count_trainable(mon_model)}")

        start = time()
        train_res = run_epoch(mon_model, train_loader, loss_fn, optimiser)
        val_res = run_epoch(mon_model, val_loader, loss_fn)
        norm_fc, norm_l4 = weight_norms(mon_model)
        row = {
            'loss/train': train_res['loss'], 'loss/val': val_res['loss'],
            'acc/train': train_res['acc'], 'acc/val': val_res['acc'],
            'gap/loss_val_moins_train': val_res['loss'] - train_res['loss'],
            'gap/acc_train_moins_val': train_res['acc'] - val_res['acc'],
            'lr/head': optimiser.param_groups[0]['lr'],
            'lr/layer4': optimiser.param_groups[1]['lr'],
            'grad_norm/train': train_res['grad_norm'],
            'weights_norm/fc': norm_fc, 'weights_norm/layer4': norm_l4,
            'time/epoch_s': time() - start,
        }
        logger.scalars(epoch, row)
        history.append({'epoch': epoch, **row})
        if epoch % cfg['hist_every'] == 0:
            logger.histograms(mon_model, epoch)

        if not math.isfinite(val_res['loss']):
            print("Loss de validation non finie (NaN) : arret.")
            break
        improved = stopper.update(val_res['loss'], mon_model, epoch)
        scheduler.step(val_res['loss'])   # le scheduler regarde la loss de validation

        print(f"Epoch {epoch:2d} | loss train {train_res['loss']:.4f} val {val_res['loss']:.4f} | "
              f"acc train {train_res['acc']:.3f} val {val_res['acc']:.3f} | "
              f"lr fc {row['lr/head']:.1e} | {row['time/epoch_s']:.1f}s"
              f"{'  <- meilleur modele' if improved else ''}")
        if stopper.should_stop:
            print(f"Arret anticipe : pas de progres depuis {stopper.patience} epochs "
                  f"(meilleur epoch : {stopper.best_epoch}).")
            break
    print(f"Duree totale : {time() - start_total:.0f} s")

    # on revient aux poids du meilleur epoch (pas ceux du dernier)
    if stopper.best_state is not None:
        mon_model.load_state_dict(stopper.best_state)
    best_row = history[stopper.best_epoch] if stopper.best_epoch >= 0 else history[-1]
    save_curves(history, run_dir)

    # ------------------------------
    # rapport sur la validation (meilleur modele)
    val_res = run_epoch(mon_model, val_loader, loss_fn)
    val_preds = val_res['probs'].argmax(axis=1)
    logger.figure('validation/matrice_de_confusion',
                  confusion_figure(val_res['labels'], val_preds, class_names, 'Validation'),
                  'confusion_validation.png')
    fig = misclassified_figure(val_files, val_res['labels'], val_preds, val_res['probs'], class_names)
    if fig is not None:
        logger.figure('validation/images_mal_classees', fig, 'mal_classees_validation.png')

    # ------------------------------
    # test final : une seule fois, apres avoir choisi le modele sur la validation
    test_res = run_epoch(mon_model, test_loader, loss_fn)
    test_preds = test_res['probs'].argmax(axis=1)
    test_report = classification_report(test_res['labels'], test_preds, target_names=class_names,
                                        digits=3, zero_division=0)
    test_report_dict = classification_report(test_res['labels'], test_preds, target_names=class_names,
                                             output_dict=True, zero_division=0)
    test_auc = safe_auc(test_res['labels'], test_res['probs'])
    test_wrong = [os.path.basename(test_files[i]) for i in range(len(test_files))
                  if test_res['labels'][i] != test_preds[i]]
    logger.figure('test/matrice_de_confusion',
                  confusion_figure(test_res['labels'], test_preds, class_names, 'Test'),
                  'confusion_test.png')
    logger.text('test/rapport', test_report)

    print("\n=== Resultat final (meilleur modele) ===")
    print(f"Meilleur epoch : {stopper.best_epoch}   (epochs effectues : {len(history)})")
    print(f"Validation : loss {best_row['loss/val']:.4f}, accuracy {best_row['acc/val']:.3f} "
          f"(train : {best_row['acc/train']:.3f})")
    print(f"Test : accuracy {test_res['acc']:.3f}, loss {test_res['loss']:.4f}, "
          f"AUC {test_auc if test_auc is None else round(test_auc, 3)}")
    print(test_report)
    print(f"Images de test mal classees ({len(test_wrong)}) : {test_wrong}")

    # ------------------------------
    # sauvegarde : modele complet (compatible avec les scripts d'evaluation) + resume du run
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'cat_model_tp3_{run_name}.pth')
    mon_model.categories = idx_to_class
    torch.save(mon_model.to('cpu'), model_path)
    print(f"Modele sauvegarde : {model_path}")
    print(f"Pour l'evaluer : MODEL_NAME = 'cat_model_tp3_{run_name}.pth' dans TP_pytorch_2_evaluation_full.py")

    results = {
        'run_name': run_name, 'config': cfg,
        'best_epoch': stopper.best_epoch, 'epochs_run': len(history),
        'stopped_early': stopper.should_stop,
        'val': {'loss': best_row['loss/val'], 'acc': best_row['acc/val'], 'acc_train': best_row['acc/train']},
        'test': {'acc': test_res['acc'], 'loss': test_res['loss'], 'auc': test_auc,
                 'report': test_report_dict, 'misclassified': test_wrong},
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.close()
    print(f"Suivi : tensorboard --logdir {runs_dir}")


if __name__ == '__main__':
    main()
