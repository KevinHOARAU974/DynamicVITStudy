# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:14:05 2025

@author: Kévin
"""
import os

import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from torch.utils.data import DataLoader

import shutil
import random
from tqdm import tqdm
from PIL import Image

def is_image_strictly_valid(path):
    try:
        with Image.open(path) as img:
            img.verify()  # Vérifie l'entête
        with Image.open(path) as img:
            img.load()    # Charge toute l'image
        return True
    except Exception:
        return False

def copy_files(files, split):
    print('Copying files to', split)

    for file in tqdm(files, disable=False):
        if not is_image_strictly_valid(file):
            print(f"Warning: Invalid image file: {file}")
            continue

        classe = os.path.basename(os.path.dirname(file))

        # Créer le dossier de destination si nécessaire
        dest_dir = os.path.join(split, classe)
        os.makedirs(dest_dir, exist_ok=True)

        # Chemin complet vers le fichier copié
        dst_path = os.path.join(dest_dir, os.path.basename(file))

        try:
            # Copier le fichier (avec métadonnées)
            shutil.copy2(file, dst_path)

            # Vérifier l'intégrité simple (par taille)
            src_size = os.path.getsize(file)
            dst_size = os.path.getsize(dst_path)

            if src_size != dst_size:
                print(f"Corrupted file (different size) : {file}")
        except Exception as e:
            print(f"error during the copy of {file} : {e}")


def create_splits(data_dir, train_pct, val_pct, max_samples=None):
    # Make sure the percentages add up to 100
    assert train_pct + val_pct <= 1.0, "Train and validation percentages should sum up to 1.0 or less"
    
    classe_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    print(classe_names)

    # Create directories for the splits if they don't exist
    for split in ['train', 'val', 'test']:
        
        if os.path.exists(split):
            shutil.rmtree(split)
        
        for classe in classe_names:
            os.makedirs(os.path.join(split, classe), exist_ok=True)

    # Gather all file names (without extensions)

    file_names = []

    for fold, subfold, files in os.walk(data_dir):
        if fold != data_dir:  # Exclut les fichiers à la racine
            for f in files:
                full_path = os.path.join(fold, f)
                file_names.append(full_path)

    # Shuffle the file names
    random.shuffle(file_names)

    # If max_samples is set, truncate the list
    if max_samples is not None:
        file_names = file_names[:max_samples]

    # Calculate split sizes
    total_files = len(file_names)
    train_size = int(total_files * train_pct)
    val_size = int(total_files * val_pct)

    # Split the file names
    train_files = file_names[:train_size]
    val_files = file_names[train_size:train_size + val_size]
    test_files = file_names[train_size + val_size:]


    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"\nDataset split complete: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test samples.")


def create_data_loaders(dataset_path, batch_size, img_size):
    # Training data transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Validation and test transform without augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    # Create datasets for each split
    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'train'),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'val'),
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'test'),
        transform=eval_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader