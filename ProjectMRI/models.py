#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Time    :   2019/04/05 01:51:09
@Author  :   Wu
@Version :   1.0
@Desc    :   The models' definitions and some training and evaluation functions 
'''

import torch
import torch.nn as nn
import torch.utils.data as Data

import math
import numpy as np
import matplotlib.pyplot as plt

# import the files of mine
from logger import log

class MyCNN(nn.Module):
    def __init__(self, image_size, num_class, img_in_channels):
        super(MyCNN, self).__init__()
        # conv1: Conv2d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=img_in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # conv2: Conv2d -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # fully connected layer
        self.fc = nn.Linear(32 * (image_size // 4) * (image_size // 4), num_class)

    def forward(self, x):
        """
        input: N * 3 * image_size * image_size\\
        output: N * num_classes
        """
        x = self.conv1(x)
        x = self.conv2(x)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output




class VGG(nn.Module):
    def __init__(self, cfg, image_size, num_classes, img_in_channels):
        super(VGG, self).__init__()
        self.img_in_channels = img_in_channels
        self.features = self._make_layers(cfg)
        # linear layer
        self.classifier = nn.Linear(256 * 512 , num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        """
        cfg: a list define layers this layer contains
            'M': MaxPool, number: Conv2d(out_channels=number) -> BN -> ReLU
        """
        layers = []
        in_channels = self.img_in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def train(model, train_loader, loss_func, optimizer, device):
    """
    train model using loss_fn and optimizer in an epoch.\\
    model: CNN networks\\
    train_loader: a Dataloader object with training data\\
    loss_func: loss function\\
    device: train on cpu or gpu device
    """
    total_loss = 0
    model.train()
    # train the model using minibatch
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # forward
        outputs = model(images)
        loss = loss_func(outputs, targets)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # every 10 iteration, print loss
        if (i + 1) % 10 == 0 or i + 1 == len(train_loader):
            log.logger.info("Step [{}/{}] Train Loss: {}".format(i+1, len(train_loader), loss.item()))
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device, test=True):
    """
    model: CNN networks\\
    val_loader: a Dataloader object with validation data\\
    device: evaluate on cpu or gpu device\\
    return classification accuracy of the model on val dataset
    """
    # evaluate the model
    model.eval()
    # context-manager that disabled gradient computation
    with torch.no_grad():
        correct = 0
        total = 0

        for _, (images, targets) in enumerate(val_loader):
            # device: cpu or gpu
            images = images.to(device)
            targets = targets.to(device)

            # predict with the model
            outputs = model(images)

            # return the maximum value of each row of the input tensor in the 
            # given dimension dim, the second return vale is the index location
            # of each maxium value found(argmax)
            _, predicted = torch.max(outputs.data, dim=1)

            correct += (predicted == targets).sum().item()

            total += targets.size(0)

        accuracy = correct / total
        log.logger.info('Accuracy on {} set is {}/{} ({:.4f}%)'.format(
            'test' if test else 'train', correct, total, 100 * accuracy))
        return accuracy


def evaluate_vote(model, val_loader, device, num_classes):
    """
    model: CNN networks\\
    val_loader: a Dataloader object with validation data\\
    device: evaluate on cpu or gpu device\\
    return classification accuracy of the model on val dataset
    """
    y_true = []
    y_pred = []
    # evaluate the model
    model.eval()
    # context-manager that disabled gradient computation
    with torch.no_grad():
        correct = 0
        total = 0

        for i, (images, targets) in enumerate(val_loader):
            # device: cpu or gpu
            images = images.to(device)
            targets = targets.to(device)

            # predict with the model
            outputs = model(images)

            # return the maximum value of each row of the input tensor in the 
            # given dimension dim, the second return vale is the index location
            # of each maxium value found(argmax)
            _, predicted = torch.max(outputs.data, dim=1)

            correct += (predicted == targets).sum().item()

            for y_true_temp in targets:
                y_true.append(int(y_true_temp.cpu().numpy()))
            for y_pred_temp in predicted:
                y_pred.append(int(y_pred_temp.cpu().numpy()))

            total += targets.size(0)

        accuracy = correct / total

        # voting
        votes = [0 for i in range(num_classes)]
        for i in y_pred:
            votes[i] += 1
        vote_result = votes.index(max(votes))

        
        log.logger.info('y_true: {}'.format(y_true))
        log.logger.info('y_pred: {}'.format(y_pred))
        log.logger.info('vote_result: {}'.format(vote_result))
        log.logger.info('Success?: {}'.format(y_true[0] == vote_result))
        log.logger.info('Accuracy is {}/{} ({:.4f}%)'.format(correct, total, 100 * accuracy))
    
        return accuracy


def show_curve(ys, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.show()


def fit(model, num_epochs, optimizer, device, train_loader, test_loader):
    """
    train and evaluate an classifier num_epochs times.\\
    We use optimizer and cross entropy loss to train the model. \\
    Args: \\
        model: CNN network\\
        num_epochs: the number of training epochs\\
        optimizer: optimize the loss function
    """

    # loss and optimizer
    loss_func = nn.CrossEntropyLoss()

    model.to(device)
    loss_func.to(device)

    # log train loss and test accuracy
    losses = []
    accs = []

    for epoch in range(num_epochs):
        log.logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # train step
        loss = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)

        # evaluate step
        train_accuracy = evaluate(model, train_loader, device, test=False)
        test_accuracy = evaluate(model, test_loader, device, test=True)
        accs.append(test_accuracy)

    # show curve
    show_curve(losses, "train loss")
    show_curve(accs, "test accuracy")