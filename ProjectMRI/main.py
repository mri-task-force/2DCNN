#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2019/04/05 00:44:03
@Author  :   Wu
@Version :   2.0
@Desc    :   main python file of the project
'''

import torch
import torch.utils.data as Data
import os

# import the files of mine
from logger import log, tensorboard_dir
from models.vgg import VGG, cfg
import utility.save_load
import utility.fitting
import process.load_dataset_v3
import models.resnets
import utility.evaluation

# remove the tensorboard_dir before running
try:
    os.system('rm -r {}'.format(tensorboard_dir))
except Exception as e:
    print(e)

# Device configuration, cpu, cuda:0/1/2/3 available
device = torch.device('cuda:6')

# Hyper parameters
batch_size = 100
num_epochs = 50
lr = 0.001
num_classes = 3 
model_name = 'ResNet34-0'

# Log the preset parameters and hyper parameters
log.logger.info("Preset parameters:")
log.logger.info('num_classes: {}'.format(num_classes))
log.logger.info('model_name: {}'.format(model_name))
log.logger.info('device: {}'.format(device))
log.logger.info("Hyper parameters:")
log.logger.info('batch_size: {}'.format(batch_size))
log.logger.info('num_epochs: {}'.format(num_epochs))
log.logger.info('lr: {}'.format(lr))


# declare and define an objet of the model
model = models.resnets.resnet34(num_classes=num_classes, img_in_channels=1)

log.logger.info(model)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=0.0001
)

train_data, test_data, sampler = process.load_dataset_v3.load_dataset(isCut=False, data_choose=2)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, sampler=sampler)
train_loader_eval = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


try:
    log.logger.critical('Start training')
    utility.fitting.fit(model, num_epochs, optimizer, device, train_loader, test_loader, train_loader_eval, num_classes)
except KeyboardInterrupt as e:
    log.logger.error('KeyboardInterrupt: {}'.format(e))
except Exception as e:
    log.logger.error('Exception: {}'.format(e))
finally:
    log.logger.info("Train finished")
    utility.save_load.save_model(
        model=model,
        path='./trained_model/{}.pt'.format(model_name)
    )
    model = utility.save_load.load_model(
        model=models.resnets.resnet34(num_classes=num_classes, img_in_channels=1),
        path='./trained_model/{}.pt'.format(model_name),
        device=device
    )
    utility.evaluation.evaluate(model=model, val_loader=train_loader_eval, device=device, num_classes=3, test=False)
    utility.evaluation.evaluate(model=model, val_loader=test_loader, device=device, num_classes=3, test=True)
    log.logger.info('Finished')


# print('type(train_loader):', type(train_loader))
# dataiter = iter(train_loader)
# images, targets = dataiter.next()
# print(type(images), type(targets))
# print(images.size(), targets.size())


# def load_model_and_test():
#     """
#     load the saved model, and take patients one by one to test
#     """
#     new_model = process.read_data.load_model(
#         model=VGG(cfg[vgg_type], image_size=image_size, num_classes=num_classes, img_in_channels=1),
#         model_path='./trained_model/{}.pt'.format(model_name),
#         device=device
#     )

#     def get_patient_DataLoader(patient_files, patient_label):
#         """
#         return a `DataLoader` object, the slides of a particular patient
#         """
#         patient_labels = [patient_label for x in patient_files]
#         patient_test_data = MyDataset(loader=default_loader, files=patient_files, labels=patient_labels)
#         patient_test_loader = torch.utils.data.DataLoader(dataset=patient_test_data, batch_size=batch_size, shuffle=False)
#         return patient_test_loader

#     for i in range(len(paths_each_patient)):
#         if (len(paths_each_patient[i]) == 0):
#             continue
#         patient_test_loader = get_patient_DataLoader(patient_files=paths_each_patient[i], patient_label=patient_labels[i])
#         utility.evaluation.evaluate_vote(model=new_model, val_loader=patient_test_loader, device=device, num_classes=num_classes)

# load_model_and_test()