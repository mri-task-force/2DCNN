#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fitting.py
@Time    :   2019/04/13 19:20:40
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import tensorflow as tf

# import the files of mine
from logger import log, tensorboard_dir
import utility.evaluation



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

        # count the number of samples of each class in a batch
        # class_sample_num = {}
        # y_true = [int(x.cpu().numpy()) for x in targets]
        # for i in range(len(y_true)):
        #     if y_true[i] not in class_sample_num:
        #         class_sample_num[y_true[i]] = 1
        #     else:
        #         class_sample_num[y_true[i]] += 1
        # log.logger.info('class_sample_num: {}'.format(class_sample_num))

    return total_loss / len(train_loader)


def fit(model, num_epochs, optimizer, device, train_loader, test_loader, train_loader_eval, num_classes):
    """
    train and evaluate an classifier num_epochs times.\\
    We use optimizer and cross entropy loss to train the model. \\
    Args:
        model: CNN network
        num_epochs: the number of training epochs
        optimizer: optimize the loss function
        device: the device to train
        train_loader: train data loader for training
        test_loader: test data loader for evaluation
        train_loader_eval: train data loader for evaluation
        num_classes: the number of classes
    """

    # loss and optimizer
    loss_func = nn.CrossEntropyLoss()

    model.to(device)
    loss_func.to(device)

    # log train loss and test accuracy
    losses = []
    train_accs = []
    test_accs = []
    train_class_accs = [[],[],[]]
    test_class_accs = [[],[],[]]
    
    ######### tensorboard #########
    writer_acc = [tf.summary.FileWriter(tensorboard_dir + '/train/'), tf.summary.FileWriter(tensorboard_dir + '/test/')]
    writer_train_class = [tf.summary.FileWriter(tensorboard_dir + '/train_class{}/'.format(i)) for i in range(3)]
    writer_test_class = [tf.summary.FileWriter(tensorboard_dir + '/test_class{}/'.format(i)) for i in range(3)]

    log_var = [tf.Variable(0.0) for i in range(3)]
    tf.summary.scalar('acc', log_var[0])
    tf.summary.scalar('train class acc', log_var[1])
    tf.summary.scalar('test class acc', log_var[2])

    write_op = tf.summary.merge_all()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    ######### tensorboard #########

    for epoch in range(num_epochs):
        log.logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # train step
        loss = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)

        # evaluate step
        train_accuracy, train_confusion, train_class_acc = utility.evaluation.evaluate(model, train_loader_eval, device, num_classes, test=False)
        test_accuracy, test_confusion, test_class_acc = utility.evaluation.evaluate(model, test_loader, device, num_classes, test=True)

        ######### tensorboard #########
        accs = [train_accuracy, test_accuracy]
        for iw, w in enumerate(writer_acc):
            summary = session.run(write_op, {log_var[0]: accs[iw]})
            w.add_summary(summary, epoch)
            w.flush()

        for iw, w in enumerate(writer_train_class):
            summary = session.run(write_op, {log_var[1]: float(train_confusion[iw, -1])})
            w.add_summary(summary, epoch)
            w.flush()

        for iw, w in enumerate(writer_test_class):
            summary = session.run(write_op, {log_var[2]: float(test_confusion[iw, -1])})
            w.add_summary(summary, epoch)
            w.flush()
        ######### tensorboard #########

        # with SummaryWriter(log_dir=tensorboard_dir, comment='train') as writer:
        #     writer.add_scalar('data/train_accuracy', train_accuracy, epoch)
        #     writer.add_scalar('data/test_accuracy', test_accuracy, epoch)
        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        train_class_accs[0].append(train_class_acc[0])
        train_class_accs[1].append(train_class_acc[1])
        train_class_accs[2].append(train_class_acc[2])
        test_class_accs[0].append(test_class_acc[0])
        test_class_accs[1].append(test_class_acc[1])
        test_class_accs[2].append(test_class_acc[2])

    utility.evaluation.show_curve(train_accs, test_accs,"acc")
    utility.evaluation.show_curve_v2(
        train_class_accs[0], train_class_accs[1], train_class_accs[2], "train classes acc")
    utility.evaluation.show_curve_v2(
        test_class_accs[0], test_class_accs[1], test_class_accs[2], "test classes acc")