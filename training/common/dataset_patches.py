import os
import torch.utils.data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from random import sample, shuffle


def get_default_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform


def get_default_augmented_transform():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.95, 1.05))], 0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def preprocess_image(path, transform, augmentation):
    image = Image.open(path)
    image = transform(image)
    return image


def preprocess_image_default(path, augmentation):
    transform = get_default_augmented_transform() if augmentation else get_default_transform()
    return preprocess_image(path, transform, augmentation)


def get_preview_of_preprocessed_image(path):
    image = Image.open(path)
    return image


def preprocessing_description():
    return "Random rotation, translation and scale: degrees=90, translate=(0.1, 0.1), scale=(0.95, 1.05)"


class DDSM(torch.utils.data.Dataset):
    def __init__(self, root, image_list_path, transform, augmentation):
        self.root = root

        with open(image_list_path, 'r') as f:
            self.images = [(line.strip().split(' ')[0], int(line.strip().split(' ')[1])) for line in f.readlines()]

        self.image_names = [filename for filename, ground_truth in self.images]
        self.normal_indices = [i for i, item in enumerate(self.images) if item[1] == 0]
        self.benign_indices = [i for i, item in enumerate(self.images) if item[1] == 1]
        self.cancer_indices = [i for i, item in enumerate(self.images) if item[1] == 2]
        self.shuffled_indices = None
        self.pick_new_normal_images()

        self.transform = transform
        self.augmentation = augmentation

        classes, class_count = np.unique([self.images[i][1] for i in self.shuffled_indices], return_counts=True)
        if (classes != [0, 1, 2]).all():
            raise RuntimeError("DDSM Dataset: classes are missing or in wrong order")
        self.weight = 1 / (class_count / np.amin(class_count))

        print("Dataset balance (normal, benign, malignant):", class_count)
        print(preprocessing_description())
        print("Augmentation:", augmentation)

    def pick_new_normal_images(self):
        self.shuffled_indices = self.benign_indices + \
                                self.cancer_indices + \
                                sample(self.normal_indices, min(len(self.benign_indices), len(self.cancer_indices)))
        shuffle(self.shuffled_indices)

    @staticmethod
    def create_patch_dataset(split):
        patch_dir = '../data/ddsm_3class'
        image_list = '../data/ddsm_3class/' + split + '.txt'
        transform = get_default_augmented_transform() if split == 'train' else get_default_transform()
        dataset = DDSM(patch_dir, image_list, transform, augmentation=split == 'train')
        return dataset

    def __len__(self):
        return len(self.shuffled_indices)

    def __getitem__(self, idx):
        image_name, ground_truth = self.images[self.shuffled_indices[idx]]
        image = preprocess_image(os.path.join(self.root, image_name), self.transform, self.augmentation)
        return image, ground_truth
