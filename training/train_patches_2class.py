# classification into (normal | suspicious) on patches

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torchnet
from torch.utils.data.sampler import SubsetRandomSampler
from munch import Munch
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from common.dataset_patches import DDSM
from common.model import get_resnet_model


def accuracy(output, target):
    pred = output.max(1)[1]
    return 100.0 * target.eq(pred).float().mean()


def save_checkpoint(checkpoint_dir, state, epoch, loss, accuracy):
    file_path = os.path.join(checkpoint_dir, 'checkpoint_{:08d}_{:.4f}_{:.4f}.pth.tar'.format(epoch, loss, accuracy))
    torch.save(state, file_path)
    return file_path


def adjust_learning_rate(optimizer, epoch):
    lr = cfg.optimizer.lr
    for e in cfg.optimizer.lr_decay_epochs:
        if epoch >= e:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    auc0 = torchnet.meter.AUCMeter()
    auc1 = torchnet.meter.AUCMeter()
    model.train()
    end = time.time()
    for i, (input_data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = Variable(input_data)
        target_var = Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        acc = accuracy(output.data, target)
        losses.update(loss.item(), input_data.size(0))
        accuracies.update(acc, input_data.size(0))
        prob = nn.Softmax(dim=1)(output)
        auc0.add(prob.data[:, 0], target.eq(0))
        auc1.add(prob.data[:, 1], target.eq(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.training.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, accuracy=accuracies))

    return batch_time.avg, data_time.avg, losses.avg, accuracies.avg, auc0.value()[0], auc1.value()[0]


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    auc0 = torchnet.meter.AUCMeter()
    auc1 = torchnet.meter.AUCMeter()
    model.eval()
    end = time.time()

    for i, (input_data, target) in enumerate(val_loader):

        with torch.no_grad():
            target = target.cuda(async=True)
            input_var = Variable(input_data)
            target_var = Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

        acc = accuracy(output.data, target)
        losses.update(loss.item(), input_data.size(0))
        accuracies.update(acc, input_data.size(0))
        prob = nn.Softmax(dim=1)(output)
        auc0.add(prob.data[:, 0], target.eq(0))
        auc1.add(prob.data[:, 1], target.eq(1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.training.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, accuracy=accuracies))

    return batch_time.avg, losses.avg, accuracies.avg, auc0.value()[0], auc1.value()[0]


def main():
    if cfg.training.resume is not None:
        log_dir = cfg.training.log_dir
        checkpoint_dir = os.path.dirname(cfg.training.resume)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        log_dir = os.path.join(cfg.training.logs_dir, '{}_{}'.format(timestamp, cfg.training.experiment_name))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        checkpoint_dir = os.path.join(cfg.training.checkpoints_dir,
                                      '{}_{}'.format(timestamp, cfg.training.experiment_name))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    print('log_dir: {}'.format(log_dir))
    print('checkpoint_dir: {}'.format(checkpoint_dir))

    checkpoint_path = None
    if cfg.training.resume is not None:
        if os.path.isfile(cfg.training.resume):
            checkpoint_path = cfg.training.resume
        else:
            print("=> no checkpoint found at '{}'".format(cfg.training.resume))
            print('')
            raise Exception

    model, start_epoch, optimizer_state, features_layer = get_resnet_model(checkpoint_path, subtype=cfg.arch.model, classes=2)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.optimizer.lr,
                                momentum=cfg.optimizer.momentum,
                                weight_decay=cfg.optimizer.weight_decay)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    train_dataset = DDSM.create_patch_dataset('train')
    val_dataset = DDSM.create_patch_dataset('val')

    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(train_dataset.weight).float()).cuda()

    if cfg.training.debug:
        # limit training data for debugging:
        train_indices = range(cfg.data.batch_size * 10)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.workers, pin_memory=True, sampler=SubsetRandomSampler(train_indices))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.workers, pin_memory=True, sampler=SubsetRandomSampler(train_indices))
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
            num_workers=cfg.data.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.workers, pin_memory=True)

    train_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    best_loss = 1e15
    wait = 0
    min_delta = cfg.training.early_stopping_mindelta
    patience = cfg.training.early_stopping_patience
    for epoch in range(start_epoch, cfg.training.epochs):
        lr = adjust_learning_rate(optimizer, epoch)
        train_summary_writer.add_scalar('learning_rate', lr, epoch + 1)

        train_dataset.pick_new_normal_images()
        train_batch_time, train_data_time, train_loss, train_accuracy, train_auc0, train_auc1 = train(
            train_loader, model, criterion, optimizer, epoch)
        train_summary_writer.add_scalar('batch_time', train_batch_time, epoch + 1)
        train_summary_writer.add_scalar('loss', train_loss, epoch + 1)
        train_summary_writer.add_scalar('accuracy', train_accuracy, epoch + 1)
        train_summary_writer.add_scalar('auc0', train_auc0, epoch + 1)
        train_summary_writer.add_scalar('auc1', train_auc1, epoch + 1)

        val_batch_time, val_loss, val_accuracy, val_auc0, val_auc1 = validate(
            val_loader, model, criterion)
        val_summary_writer.add_scalar('batch_time', val_batch_time, epoch + 1)
        val_summary_writer.add_scalar('loss', val_loss, epoch + 1)
        val_summary_writer.add_scalar('accuracy', val_accuracy, epoch + 1)
        val_summary_writer.add_scalar('auc0', val_auc0, epoch + 1)
        val_summary_writer.add_scalar('auc1', val_auc1, epoch + 1)

        # Early Stopping
        current_loss = val_loss
        if current_loss is None:
            print("current val_loss is None")
        else:
            if (current_loss - best_loss) < -min_delta:
                best_loss = current_loss
                wait = 1
            else:
                if wait >= patience:
                    print("Terminated training due to early stopping")
                    checkpoint_path = save_checkpoint(checkpoint_dir, {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch + 1, val_loss, val_accuracy)
                    cfg.training.log_dir = log_dir
                    cfg.training.resume = checkpoint_path
                    with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
                        f.write(cfg.toYAML())
                    print("Checkpoint written: " + checkpoint_path)
                    return  # stop training
                wait += 1

        if (epoch + 1) % cfg.training.checkpoint_epochs == 0:
            checkpoint_path = save_checkpoint(checkpoint_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch + 1, val_loss, val_accuracy)
            cfg.training.log_dir = log_dir
            cfg.training.resume = checkpoint_path
            with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
                f.write(cfg.toYAML())
            print("Checkpoint written: " + checkpoint_path)

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', metavar='PATH', help='path to config file')
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, 'r') as config_file:
        cfg = Munch.fromYAML(config_file)
    startTime = datetime.now()
    main()
    print("Runtime: %s" % (datetime.now() - startTime))
