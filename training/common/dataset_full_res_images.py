import os
import torch.utils.data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def get_default_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform


def preprocess_image(path, transform, augmentation):
    image = Image.open(path).convert('RGB')
    image = transform(image)
    return image


def preprocess_image_default(path, augmentation):
    transform = get_default_transform()
    return preprocess_image(path, transform, augmentation)


def get_preview_of_preprocessed_image(path):
    image = Image.open(path)
    return image


def preprocessing_description():
    return "No preprocessing (full res images)"


class DDSM(torch.utils.data.Dataset):
    def __init__(self, root, image_list_path, transform, augmentation):
        self.root = root
        name2class = {
            'normal': 0,
            'benign': 1,
            'cancer': 2,
        }
        with open(image_list_path, 'r') as f:
            self.images = [(line.strip(), name2class[line.strip()[:6]]) for line in f.readlines()]
        self.image_names = [filename for filename, ground_truth in self.images]
        self.transform = transform
        self.augmentation = augmentation

        classes, class_count = np.unique([label for _, label in self.images], return_counts=True)
        if (classes != [0, 1, 2]).all():
            raise RuntimeError("DDSM Dataset: classes are missing or in wrong order")
        self.weight = 1 / (class_count / np.amin(class_count))

        print("Dataset balance (normal, benign, malignant):", class_count)
        print(preprocessing_description())
        print("Augmentation:", augmentation)

    @staticmethod
    def create_full_image_dataset(split):
        raw_image_dir = '../data/ddsm_raw'
        image_list = '../data/ddsm_raw_image_lists/' + split + '.txt'
        dataset = DDSM(raw_image_dir, image_list, get_default_transform(), augmentation=False)
        return dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name, ground_truth = self.images[idx]
        image = preprocess_image(os.path.join(self.root, image_name), self.transform, augmentation=False)
        return image, ground_truth
