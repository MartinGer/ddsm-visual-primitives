import os
from common.model import get_resnet152_3class_model

def get_class_influences_for_class(checkpoint_path):
    model, _, _, _ = get_resnet152_3class_model(checkpoint_path)

    class_influence_weights = {
        '0': enumerate(model._modules['module'].fc.weight.data[0], 1),
        '1': enumerate(model._modules['module'].fc.weight.data[1], 1),
        '2': enumerate(model._modules['module'].fc.weight.data[2], 1)
    }
    class_influence_weights_sorted = {
        '0': sorted(class_influence_weights['0'], key=lambda x:x[1]),
        '1': sorted(class_influence_weights['1'], key=lambda x:x[1]),
        '2': sorted(class_influence_weights['2'], key=lambda x:x[1])
    }

    return class_influence_weights_sorted[0], class_influence_weights_sorted[1], class_influence_weights_sorted[2]
