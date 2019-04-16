#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2019/04/13 19:25:09
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import copy

# import the files of mine
from logger import log


def evaluate(model, val_loader, device, num_classes, test=True):
    """
    evaluate the model\\
    Args: 
        model: CNN networks
        val_loader: a Dataloader object with validation data
        device: evaluate on cpu or gpu device
        num_classes: the number of classes
    Return: 
        classification accuracy of the model on val dataset
    """

    confusion_matrix = np.zeros((num_classes, num_classes + 2))
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

            # count the class_correct and class_total for each class
            y_true = [int(x.cpu().numpy()) for x in targets]
            y_pred = [int(x.cpu().numpy()) for x in predicted]
            for i in range(len(y_true)):
                confusion_matrix[y_true[i], y_pred[i]] += 1

    
        # calculate accuracy
        accuracy = correct / total
        # accuracy for each class
        for i in range(num_classes):
            confusion_matrix[i, -2] = np.sum(confusion_matrix[i, :num_classes])
            confusion_matrix[i, -1] = confusion_matrix[i, i] / confusion_matrix[i, -2]

        log.logger.info('Accuracy on {} set is {}/{} ({:.4f}%)'.format(
            'test ' if test else 'train', correct, total, 100 * accuracy))

        class_acc = []
        for i in range(num_classes):
            log.logger.info('Confusion Matrix on {} set (class {}): {:5d} {:5d} {:5d}    Acc: {}/{} ({:.4f}%)'.format(
                'test ' if test else 'train', i, 
                int(confusion_matrix[i, 0]), 
                int(confusion_matrix[i, 1]), 
                int(confusion_matrix[i, 2]),
                int(confusion_matrix[i, i]),
                int(confusion_matrix[i, -2]), 
                100 * float(confusion_matrix[i, -1])
            ))
            class_acc.append(float(confusion_matrix[i, -1]))  

        return accuracy, confusion_matrix, class_acc


def _evaluate_vote(model, val_loader, device, num_classes):
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


def show_curve(y1s, y2s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    plt.plot(x, y1, c='b', label='train') # train
    plt.plot(x, y2, c='r', label='test') # test
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    #plt.savefig("./../pics/{}.png".format(title))
    plt.savefig("{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')


def show_curve_v2(y1s, y2s, y3s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    y3 = np.array(y3s)
    plt.plot(x, y1, c='r', label='class0')  # class0
    plt.plot(x, y2, c='g', label='class1')  # class1
    plt.plot(x, y3, c='b', label='class1')  # class2
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    #plt.savefig("./../pics/{}.png".format(title))
    plt.savefig("{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')
