"""Reduit la taille des images du dataset pour alleger le depot.

Le ResNet-18 redimensionne tout en 224 x 224 : des photos de 4000 x 3000 pixels
(plusieurs Mo) ne servent a rien. On les ramene a 512 pixels maximum sur le grand
cote (on voit toujours tres bien ce que la photo represente).

Les noms de fichiers et les formats (.jpg / .png) ne changent pas.
Avant d'ecraser une image, l'original est copie dans le dossier de sauvegarde
(hors depot). Une sauvegarde existante n'est jamais ecrasee : on peut donc relancer
le script sans perdre les vrais originaux.

Usage :
    python tools/resize_images.py                 # simulation : n'ecrit rien
    python tools/resize_images.py --apply         # reduit les images pour de vrai
    python tools/resize_images.py --apply --max-size 640 --quality 90
"""
import argparse
import io
import os
import shutil

from PIL import Image, ImageOps

# dossiers a traiter, relatifs a la racine du projet
DATASET_DIRS = ['images', 'data/test_images']
EXTENSIONS = ('.jpg', '.jpeg', '.png')
DEFAULT_BACKUP_DIRNAME = 'cats_detector_originaux'  # cree a cote du projet (hors depot) si --backup-dir est absent


def list_images(root):
    files = []
    for rel_dir in DATASET_DIRS:
        for dir_path, _, names in os.walk(os.path.join(root, rel_dir)):
            for name in sorted(names):
                if name.lower().endswith(EXTENSIONS):
                    files.append(os.path.join(dir_path, name))
    return sorted(files)


def encode(img, extension, quality):
    # encode l'image en memoire : permet de connaitre la taille finale avant d'ecrire
    buffer = io.BytesIO()
    if extension in ('.jpg', '.jpeg'):
        img.convert('RGB').save(buffer, format='JPEG', quality=quality, optimize=True)
    else:
        img.save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true', help="ecrit vraiment les images (sinon simulation)")
    parser.add_argument('--max-size', type=int, default=512, help="taille max du grand cote en pixels (defaut 512)")
    parser.add_argument('--quality', type=int, default=85, help="qualite JPEG (defaut 85)")
    parser.add_argument('--backup-dir', default=None,
                        help=f"ou sauvegarder les originaux (defaut : ../{DEFAULT_BACKUP_DIRNAME}, a cote du projet)")
    parser.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir),
                        help="racine du projet (defaut : dossier parent de tools/)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if args.backup_dir is None:
        args.backup_dir = os.path.join(os.path.dirname(root), DEFAULT_BACKUP_DIRNAME)
    print(f"Mode : {'APPLICATION' if args.apply else 'SIMULATION (rien n ecrit, ajouter --apply)'}")
    print(f"Racine : {root} | taille max : {args.max_size}px | qualite JPEG : {args.quality}")
    print(f"Sauvegarde des originaux : {args.backup_dir}\n")

    total_before = total_after = 0
    nb_resized = nb_skipped = 0

    for path in list_images(root):
        rel_path = os.path.relpath(path, root)
        extension = os.path.splitext(path)[1].lower()
        size_before = os.path.getsize(path)
        total_before += size_before

        img = Image.open(path)
        # exif_transpose : applique l'orientation (photos de telephone) avant de reduire,
        # sinon l'image pourrait finir couchee une fois les metadonnees perdues
        img = ImageOps.exif_transpose(img)
        width, height = img.size

        if max(width, height) <= args.max_size:
            nb_skipped += 1
            total_after += size_before
            print(f"  deja petite  {rel_path:<42} {width}x{height}  {size_before / 1024:7.0f} Ko")
            continue

        img.thumbnail((args.max_size, args.max_size), Image.LANCZOS)  # garde le rapport largeur / hauteur
        data = encode(img, extension, args.quality)
        total_after += len(data)
        nb_resized += 1
        print(f"  reduite      {rel_path:<42} {width}x{height} -> {img.width}x{img.height}  "
              f"{size_before / 1024:7.0f} Ko -> {len(data) / 1024:5.0f} Ko")

        if args.apply:
            backup_path = os.path.join(args.backup_dir, rel_path)
            if not os.path.exists(backup_path):  # on ne remplace jamais une sauvegarde existante
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copy2(path, backup_path)
            with open(path, 'wb') as f:
                f.write(data)

    print(f"\n{nb_resized} image(s) reduite(s), {nb_skipped} deja petite(s)")
    print(f"Taille totale : {total_before / 1e6:.1f} Mo -> {total_after / 1e6:.1f} Mo")
    if not args.apply:
        print("Simulation seulement : relancer avec --apply pour reduire les images.")


if __name__ == '__main__':
    main()
