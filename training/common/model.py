import os

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn

import sys
sys.path.insert(0, '..')
from training.models.resnet import resnet152, resnet18


def get_resnet_model(checkpoint_path=None, subtype='resnet152', classes=3):
    if subtype == 'resnet18':
        print("=> creating model 'resnet18'")
        model = resnet18(pretrained=not checkpoint_path)
        model.fc = nn.Linear(512, classes)
    else:
        print("=> creating model 'resnet152'")
        model = resnet152(pretrained=not checkpoint_path)
        model.fc = nn.Linear(2048, classes)
    features_layer = model.layer4

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    epoch = 0
    optimizer_state = None
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        optimizer_state = checkpoint['optimizer']
        epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    else:
        print("=> no checkpoint loaded, only ImageNet weights")

    return model, epoch, optimizer_state, features_layer


def get_model_from_config(cfg, epoch=None):
    if epoch:
        resume_path = cfg.training.resume.replace(cfg.training.resume[-16:-8], '{:08}'.format(epoch))
    else:
        resume_path = cfg.training.resume
    resume_path = os.path.join('../training', resume_path)
    model, epoch, optimizer_state, features_layer = get_resnet_model(resume_path, subtype=cfg.arch.model, classes=cfg.arch.num_classes)
    return model, features_layer, resume_path
