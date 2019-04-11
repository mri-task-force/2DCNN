#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utility.py
@Time    :   2019/04/05 00:41:37
@Author  :   Wu
@Version :   1.0
@Desc    :   Some utility functions
'''

import os
import SimpleITK as sitk
import numpy as np
import torch

# import the files of mine
import models
from logger import log

def get_image_paths(img_dir):
    """
    return a `list`, the paths of the images (slides) in `img_dir`\\
    Args: \\
        img_dir: the directory of the images (slides)\\
    """
    # if there is no such directory, return an empty list. 
    # Notice: Some directories are in the excel file, while not exist in the dataset folder.
    if not os.path.exists(img_dir): 
        return []
    paths = []
    for img_name in os.listdir(img_dir):
        # check the file type
        file_type = os.path.splitext(img_name)[1]
        if (not(file_type == '.dcm')):
            log.logger.warning("File is not .dcm")
            continue
        paths.append(os.path.join(img_dir, img_name))
    return sorted(paths)


def read_image(img_path):
    """
    read an image from `img_path` and return its numpy image\\
    Args: \\
        img_path: the path of the image\\
    """
    sitk_image = sitk.ReadImage(img_path)
    np_image = sitk.GetArrayFromImage(sitk_image)
    return np_image[0].astype(np.float32)


def save_model(save_path, model_to_save, temp_model, device, val_loader):
    """
    Save the model `model_to_save` to `save_path`\\
    Args: \\
        save_path: The path of the model to be saved\\
        model_to_save: The model to save\\
        temp_model: A temp model loaded from file, to test the success of saving the model\\
        device: torch.device, cpu, cuda:0/1/2/3 available\\
        val_loader: The dataloader to be evaluate by the temp_model
    """
    # save model
    torch.save(model_to_save.state_dict(), save_path)
    log.logger.info("Model has been saved at: '{}'".format(save_path))
    # A new model loaded from a file, to test the success of saving the train model
    # load model
    log.logger.info("Load the model and evaluate")
    saved_params = torch.load(save_path)
    temp_model.load_state_dict(saved_params)
    temp_model = temp_model.to(device)

    # evaluate the loaded model
    new_acc = models.evaluate(model=temp_model, val_loader=val_loader, device=device)
    log.logger.info("Accuracy of the loaded model: {}".format(new_acc))


def load_model(model, model_path, device):
    saved_params = torch.load(model_path)
    model.load_state_dict(saved_params)
    model = model.to(device)
    return model

