import os
import torch.utils.data
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
from random import random


IMAGE_SIZE_TO_ANALYZE = 1024
TARGET_ASPECT_RATIO = 2.3 / 3
TOP_CROP = 200
BOTTOM_CROP = 200
LEFT_CROP = 50
RIGHT_CROP = 50
CROP_VARIATION = 50
BLACK_LEVEL = 60
MAX_ROTATION = 15


def get_default_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform


def resize_and_pad_image(image, target_size, target_aspect_ratio, augmentation):

    d = int(random() * CROP_VARIATION) if augmentation else 0
    image = image.crop((LEFT_CROP + d, TOP_CROP + d, image.size[0] - (RIGHT_CROP + d), image.size[1] - (BOTTOM_CROP + d)))
    target_width = int(target_size * target_aspect_ratio)
    target_height = target_size

    image_ratio = image.size[0] / image.size[1]

    if target_aspect_ratio < image_ratio:
        # limit is width
        scale_ratio = target_width / image.size[0]
    else:
        # limit is height
        scale_ratio = target_height / image.size[1]

    new_size = (int(scale_ratio * image.size[0]), int(scale_ratio * image.size[1]))
    image = image.resize(new_size, resample=Image.BILINEAR)  # image shape is now (~1500, 896)
    delta_w = target_width - image.size[0]
    delta_h = target_height - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    image = ImageOps.expand(image, padding)
    angle = int((random() * 2 - 1) * MAX_ROTATION) if augmentation else 0
    image = image.rotate(angle, resample=Image.BICUBIC)
    return image


def remove_background_noise(image, augmentation):
    image = np.asarray(image).astype(np.int16)
    black_level = int(random() * BLACK_LEVEL) if augmentation else 0
    image = np.clip((image - black_level) * (255.0 / (255 - black_level)), 0, 255).astype(np.uint8)
    return image


def preprocess_image(path, target_size, transform, augmentation):
    image = Image.open(path)
    image = resize_and_pad_image(image, target_size, TARGET_ASPECT_RATIO, augmentation)
    image = remove_background_noise(image, augmentation)
    image = np.broadcast_to(np.expand_dims(image, 2), image.shape + (3,))  # image shape is now (~1500, 896, 3)
    image = transform(image)  # image shape is now (3, ~1500, 896) and a it is a tensor
    return image


def preprocess_image_default(path, augmentation):
    transform = get_default_transform()
    return preprocess_image(path, IMAGE_SIZE_TO_ANALYZE, transform, augmentation)


def get_preview_of_preprocessed_image(path):
    image = Image.open(path)
    image = resize_and_pad_image(image, IMAGE_SIZE_TO_ANALYZE, TARGET_ASPECT_RATIO, augmentation=False)
    image = remove_background_noise(image, augmentation=False)
    image = Image.fromarray(image)
    return image


def no_preprocessing(path):
    image = np.array(Image.open(path))
    image = np.broadcast_to(np.expand_dims(image, 2), image.shape + (3,))  # image shape is now (~1500, 896, 3)
    transform = get_default_transform()
    image = transform(image)  # image shape is now (3, height, width) and it is a tensor
    return image


def preprocessing_description():
    return "Preprocessing (IMAGE_SIZE_TO_ANALYZE, TARGET_ASPECT_RATIO, TOP_CROP, BOTTOM_CROP, LEFT_CROP, RIGHT_CROP, CROP_VARIATION, BLACK_LEVEL, MAX_ROTATION):\n" + \
        str((IMAGE_SIZE_TO_ANALYZE, TARGET_ASPECT_RATIO, TOP_CROP, BOTTOM_CROP, LEFT_CROP, RIGHT_CROP, CROP_VARIATION, BLACK_LEVEL, MAX_ROTATION))


def get_ground_truth_from_filename(filename):
    name2class = {
        'normal': 0,
        'benign': 1,
        'cancer': 2,
    }
    return name2class[filename[:6]]


class DDSM(torch.utils.data.Dataset):
    def __init__(self, root, image_list_path, target_size, transform, augmentation):
        self.root = root
        with open(image_list_path, 'r') as f:
            self.images = [(line.strip(), get_ground_truth_from_filename(line.strip())) for line in f.readlines()]
        self.image_names = [filename for filename, ground_truth in self.images]
        self.target_size = target_size
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
        dataset = DDSM(raw_image_dir, image_list, IMAGE_SIZE_TO_ANALYZE, get_default_transform(), split == 'train')
        return dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name, ground_truth = self.images[idx]
        image = preprocess_image(os.path.join(self.root, image_name), self.target_size, self.transform, self.augmentation)
        return image, ground_truth
