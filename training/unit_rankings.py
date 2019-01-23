from common.model import get_resnet_3class_model


def get_class_influences_for_class(checkpoint_path):
    '''
    Loads a checkpoint for a resnet152_3class architecture and returns last hidden layer unit to class activation
    weights. These can be used as an influence ranking of all units for each individual outcome class.

    :param checkpoint_path: path to a checkpoint of a resnet 152_3class network
    :return: Three, one for each class outcome, rankings (sorted lists) of (unit, weight)-pairs in sorted order
    '''
    model, _, _, _ = get_resnet_3class_model(checkpoint_path)
    # create single image analysis object

    class_influence_weights = {
        0: enumerate(model._modules['module'].fc.weight.data[0], 1),
        1: enumerate(model._modules['module'].fc.weight.data[1], 1),
        2: enumerate(model._modules['module'].fc.weight.data[2], 1)
    }
    class_influence_weights_sorted = {
        0: sorted(class_influence_weights[0], key=lambda x:-x[1]),
        1: sorted(class_influence_weights[1], key=lambda x:-x[1]),
        2: sorted(class_influence_weights[2], key=lambda x:-x[1])
    }

    return class_influence_weights_sorted[0], class_influence_weights_sorted[1], class_influence_weights_sorted[2]
