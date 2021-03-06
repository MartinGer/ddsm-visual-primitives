import argparse
import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from PIL import Image
from munch import Munch
from torch.autograd import Variable
from training.common.dataset import preprocess_image_default, no_preprocessing
from training.common.dataset_patches import preprocess_image_default as preprocess_patch_default
from training.common.model import get_resnet_3class_model

import sys
sys.path.insert(0, '..')


class AnalysisResult(object):
    def __init__(self, image_path, checkpoint_path, model):
        self.image_path = image_path
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.feature_maps = []
        self.feature_maps_maximums = []
        self.class_probs = []
        self.classification = None

    def get_top_units(self, diagnosis_class, number_of_units):
        params = list(self.model.parameters())
        weight_softmax = params[-2].data.cpu().numpy()

        # calculate influence of units
        feature_maps_maximums = np.expand_dims(self.feature_maps_maximums, 0)  # shape: (1, 2048)
        weighted_max_activations = feature_maps_maximums * weight_softmax  # shape: (num_classes=3, 2048)

        units_and_activations = []

        for unit_id, influence_per_class in enumerate(weighted_max_activations.T):  # 2048, number of units
            units_and_activations.append((unit_id, influence_per_class, self.feature_maps[unit_id]))

        ranked_units_and_activations = sorted(units_and_activations, key=lambda x: x[1][diagnosis_class], reverse=True)[
                                       :number_of_units]

        # entries of ranked_units_and_activations: unit index (0-based), influence_per_class[0,1,2], activation_map for the unit
        return ranked_units_and_activations


class SingleImageAnalysis(object):

    def __init__(self, checkpoint_path):
        self.model, _, _, self.features_layer = get_resnet_3class_model(checkpoint_path)
        self.checkpoint_path = checkpoint_path

    def analyze_one_image(self, image_path, preprocess=True):
        if preprocess:
            image = preprocess_image_default(image_path, augmentation=False)
        else:
            image = no_preprocessing(image_path)
        print("run image through model")
        return self._analyze(image, image_path)

    def analyze_one_patch(self, image_path):
        image = preprocess_patch_default(image_path, augmentation=False)
        print("run patch through model")
        return self._analyze(image, image_path)

    def _analyze(self, image, image_path):
        image_batch = image.unsqueeze(0)  # unsqueeze: (3, ~1500, 896) -> (1, 3, ~1500, 896)
        result = AnalysisResult(image_path, self.checkpoint_path, self.model)
        # extract features and max activations

        def feature_hook(_, __, layer_output):  # args: module, input, output
            # layer_output.data.shape: (2048, ~50, 28)
            nonlocal result
            result.feature_maps = layer_output.data.cpu().numpy()[0]
            result.feature_maps_maximums = result.feature_maps.max(axis=(1, 2))  # shape: (2048)  //only the max value of one activation matrix. This allows to sort them in a list

        self.features_layer._forward_hooks.clear()
        self.features_layer.register_forward_hook(feature_hook)

        with torch.no_grad():

            input_var = Variable(image_batch)
            output = self.model(input_var)  # forward pass with hooks
            class_probs = nn.Softmax(dim=1)(output).squeeze(0)  # shape: [3], i.e. [0.9457, 0.0301, 0.0242]
            class_probs = class_probs.cpu().numpy()
            result.classification = int(np.argmax(class_probs))  # int
            result.class_probs = class_probs

        return result

    def get_model(self):
        return self.model


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path')
    parser.add_argument('--image_path')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg = Munch.fromYAML(f)

    single_image_analysis = SingleImageAnalysis(os.path.join('../training', cfg.training.resume))
    single_image_analysis.analyze_one_image(args.image_path)


if __name__ == '__main__':
    test()
