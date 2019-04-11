#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2019/04/05 00:44:03
@Author  :   Wu
@Version :   1.0
@Desc    :   main python file of the project
'''

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the files of mine
from logger import log
import models
import utility


# Choose dataset
dataset = 1    # The dataset index, small dataset: 0, 6_ROI dataset: 1

# Preset parameters
root_dir = './../../Datasets/data/'                 # The root directory of the dataset
excel_file_path = './../../Datasets/data/病例信息汇总.xlsx'
excel_sheet_name = 'Sheet1'
vgg_type = 'VGG11'          # The vgg type, used for VGG model initialization
model_name = '{}-{}'.format(vgg_type, dataset)      # The model file name, used for save model
patient_num = 39            # The number of patient (folders) in this dataset
test_num = 100              # The number of test images

if dataset == 1:
    root_dir = './../../Datasets/2019_rect_pcr_data/6_ROI/'    # The root directory of the dataset
    excel_file_path = './../../Datasets/2019_rect_pcr_data/总病例信息表.xlsx'
    excel_sheet_name = '中山六院'
    vgg_type = 'VGG11'      # The vgg type, used for VGG model initialization
    model_name = '{}-{}'.format(vgg_type, dataset)  # The model file name, used for save model
    patient_num = 347       # The number of patient (folders) in this dataset
    test_num = 1000         # The number of test images

# Device configuration, cpu, cuda:0/1/2/3 available
device = torch.device('cuda:1')

# Hyper parameters
batch_size = 32
num_epochs = 5
lr = 0.001
image_size = 512


# Log the preset parameters and hyper parameters
log.logger.critical("Preset parameters:")
log.logger.info('dataset: {}'.format(dataset))
log.logger.info('root directory of the dataset: {}'.format(root_dir))
log.logger.info('model_name: {}'.format(model_name))
log.logger.info('device: {}'.format(device))
log.logger.critical("Hyper parameters:")
log.logger.info('batch_size: {}'.format(batch_size))
log.logger.info('num_epochs: {}'.format(num_epochs))
log.logger.info('lr: {}'.format(lr))
log.logger.info('image_size: {}'.format(image_size))


# Read from the excel file
df = pd.read_excel(excel_file_path, sheet_name=excel_sheet_name) # Read the sheet (first sheet by default) , df: dataframe of pandas
data = df.ix[:patient_num - 1, ['编号', '结局']].values # Read the rows (from row 1 to patient_num, no header) 

# The identifiers of patients, e.g. 'sub001'. Take the first six characters, for there's a 'sub338（术后4周肝转移）'
patient_identifiers = [x[0:6] for x in data[:, 0]] 
# The labels (results, 0, 1, 2) for each patient. Minus 1 for each label, because the original labels start at 1
patient_labels = [int(x) - 1 for x in data[:, 1]]
# log info
log.logger.info('patient_identifiers: {}'.format(patient_identifiers))
log.logger.info('patient_labels: {}'.format(patient_labels))


# The directories of each patient, e.g. './../../Datasets/2019_rect_pcr_data/6_ROI/sub001/MRI/T2'
files_dirs = ['{}{}/MRI/T2'.format(root_dir, x) for x in patient_identifiers]

# The paths of the images for each patient, e.g. ['./../../Datasets/2019_rect_pcr_data/6_ROI/sub001/MRI/T2/IMG-0001-00001.dcm', ...]
paths_each_patient = [utility.get_image_paths(x) for x in files_dirs]
files = []      # The paths of all images (slides) in dataset, 857 files for small dataset, 6694 files for 6_ROI dataset
labels = []     # The labels for each slide
for i, t_paths in enumerate(paths_each_patient): # for each patient
    for j, t_path in enumerate(t_paths):    # for the image paths of a patient
        files.append(t_path)
        labels.append(patient_labels[i])    # The labels is the same for the same patient


files_test = files[:test_num]
files_train = files[test_num:]
labels_test = labels[:test_num]
labels_train = labels[test_num:]


normalize = transforms.Normalize(
    mean=[0.48],
    std=[0.25]
)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


def default_loader(path):
    img_np = utility.read_image(path)
    img_tensor = preprocess(img_np)
    return img_tensor


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, loader, files, labels):
        # set the paths of the images
        self.images = files
        self.target = labels
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)
    
    
train_data = MyDataset(loader=default_loader, files=files_train, labels=labels_train)
test_data = MyDataset(loader=default_loader, files=files_test, labels=labels_test)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# the number of classes
num_classes = len(set(patient_labels))  
log.logger.info('num_classes: {}'.format(num_classes))


# declare and define an objet of the model
vggnet = models.VGG(models.cfg[vgg_type], image_size=image_size, num_classes=num_classes, img_in_channels=1)
log.logger.info(vggnet)

optimizer = torch.optim.Adam(vggnet.parameters(), lr=lr)


try:
    log.logger.critical('Start training')
    models.fit(vggnet, num_epochs, optimizer, device, train_loader, test_loader)
except KeyboardInterrupt as e:
    log.logger.error('KeyboardInterrupt: {}'.format(e))
except Exception as e:
    log.logger.error('Exception: {}'.format(e))
finally:
    log.logger.info("Train finished")
    utility.save_model(
        save_path='./model/{}.pt'.format(model_name),
        model_to_save=vggnet,
        temp_model=models.VGG(models.cfg[vgg_type], image_size=image_size, num_classes=num_classes, img_in_channels=1),
        device=device,
        val_loader=test_loader
    )
    log.logger.critical('Finished')


# print('type(train_loader):', type(train_loader))
# dataiter = iter(train_loader)
# images, targets = dataiter.next()
# print(type(images), type(targets))
# print(images.size(), targets.size())


def load_model_and_test():
    """
    load the saved model, and take patients one by one to test
    """
    new_model = utility.load_model(
        model=models.VGG(models.cfg[vgg_type], image_size=image_size, num_classes=num_classes, img_in_channels=1),
        model_path='./model/{}.pt'.format(model_name),
        device=device
    )

    def get_patient_DataLoader(patient_files, patient_label):
        """
        return a `DataLoader` object, the slides of a particular patient
        """
        patient_labels = [patient_label for x in patient_files]
        patient_test_data = MyDataset(loader=default_loader, files=patient_files, labels=patient_labels)
        patient_test_loader = torch.utils.data.DataLoader(dataset=patient_test_data, batch_size=batch_size, shuffle=False)
        return patient_test_loader

    for i in range(len(paths_each_patient)):
        if (len(paths_each_patient[i]) == 0):
            continue
        patient_test_loader = get_patient_DataLoader(patient_files=paths_each_patient[i], patient_label=patient_labels[i])
        models.evaluate_vote(model=new_model, val_loader=patient_test_loader, device=device, num_classes=num_classes)

load_model_and_test()