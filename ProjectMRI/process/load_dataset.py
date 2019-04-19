# coding: utf-8
'''
    2019/3/29

    读取6_ROI中的347个病人的mri数据：sub1 ~ sub364
    数据集：(肿瘤医院，中山六院)
        T2: 512*512*N (N为帧数，不同病人的MRi帧数不同)
'''

import os
import sys
sys.path.append('../')
import pickle

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# import files of mine
from logger import log


# [39人小数据集, 651人CC_ROI, 363人6_ROI]
dataset_paths = [{'xlsx_path':'/home/share/Datasets/data/information.xlsx', 
                'sheet_name' : 'Sheet1',
                'data_path'  : '/home/share/Datasets/data/'},
                {'xlsx_path' :'/home/share/Datasets/2019_rect_pcr_data/information.xlsx',
                'sheet_name' : 0,
                'data_path'  : '/home/share/Datasets/2019_rect_pcr_data/CC_ROI/'},
                {'xlsx_path' : '/home/share/Datasets/2019_rect_pcr_data/information.xlsx',
                'sheet_name' : 1,
                'data_path' : '/home/share/Datasets/2019_rect_pcr_data/6_ROI/'}]

# 保存的pkl, 数据未经处理
raw_data_paths = ['/home/share/Datasets/pickles/raw_dataset_0', 
                  '/home/share/Datasets/pickles/raw_dataset_1',
                  '/home/share/Datasets/pickles/raw_dataset_2']

# 训练集和测试集的pkl
train_test_paths = ['/home/share/Datasets/pickles/train_test_dataset_0',
                    '/home/share/Datasets/pickles/train_test_dataset_1',
                    '/home/share/Datasets/pickles/train_test_dataset_2']


# the path of the information xlsx
#inf_xlsx_path = "/home/share/Datasets/2019_rect_pcr_data/information.xlsx"
#inf_xlsx_path = "/home/share/Datasets/data/information.xlsx"
# the path of datasets in the hospital of the patients
#dataset_dir = "/home/share/Datasets/2019_rect_pcr_data/6_ROI/"
#dataset_dir = "/home/share/Datasets/data/"


# the sub-path of mri image for each patient
mri_path = "/MRI/T2"
# the sub-path of tumor mri image for each patient
tumor_path = "/MRI/T2tumor.mha"
# mri the final path is: /home/share/Datasets/data/sub001/MRI/T2
# mha the final path is: /home/share/Datasets/data/sub001/MRI/T2tumor.mha


'''
6_ROI:
{0.4688: 78, 0.3906: 3, 0.5078: 15, 0.5469: 122, 0.5859: 8, 
0.4883: 92, 0.468800008297: 7, 0.4297: 1, 0.5273: 2, 0.546899974346: 7, 
0.625: 1, 0.507799983025: 1, 0.585900008678: 1}
'''

def _load_mri(file_path):
    '''
        @ param: p_id为str类型，表示病人的编号
        @ return: 返回这个病人的mri图像的列表(N,512,512) 
        use SimpleITK to load dcm file
        读取一个病人的T2的mri图像，图像文件格式为dcm
    '''
    Len = 280 # 512*0.5469

    img_path = file_path
    image_array = None
    new_image_array = None
    if os.path.exists(img_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(img_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        origin = image.GetOrigin()  # x, y, z
        spacing = image.GetSpacing()  # x, y, z
        
        size = np.shape(image_array)[-1] # 512
        new_size = int(Len/spacing[0])

        new_image_array = []
        for i in range(len(image_array)):
            im = Image.fromarray(image_array[i],mode="I;16")
            new_image = im.resize((new_size, new_size), Image.NEAREST) # resize
            new_image = transforms.CenterCrop((320, 320))(new_image)
            # new_image.save('./pics/test_{}.tiff'.format(i), quality=95)
            new_image_array.append(np.array(new_image))

        new_image_array = np.array(new_image_array)
    else:
        log.logger.info('Cannot find file \"' + img_path + '\"')

    return new_image_array


def _load_tumor(file_path):
    '''
        @ param: p_id为str类型，表示病人的编号
        @ return: 返回这个病人的tumor标注的mri图像的(N,512,512)
        读取一个病人的T2的mri图像的肿瘤标记图像，图像文件格式为mha
        mha为三维数组，元素为0/1 int
    '''
    img_path = file_path
    image_array = None
    if os.path.exists(img_path):
        image = sitk.ReadImage(img_path)
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        origin = image.GetOrigin()  # x, y, z
        spacing = image.GetSpacing()  # x, y, z
    else:
        log.logger.info('Cannot find file \"' + img_path + '\"')

    #for i in range(len(image_array)):
    #    plt.imsave('./pics/1_{}.png'.format(i),image_array[i])

    return image_array


def _load_inf(file_path, sheet_name=0):
    ''' 
        @ return: dict类型 {p_id: label\}
        读取information.xlsx文件，加载病人编号，标签
        for 6_ROI:
        index in excel: 0 p_id, 65 label
        先读取6_ROI
        # sub338（术后4周肝转移），手动在excel去掉括号及括号中的内容
    '''
    # patients = {} # {p_id:label }
    df = pd.read_excel(io=file_path, sheet_name=sheet_name)  # 打开excel文件

    patient_ids = df.iloc[:,0].values  # 读取编号列
    patient_labels = df[u'结局'].values # 读取结局列

    patient_ids = patient_ids[~np.isnan(patient_labels)]       # 删掉 nan
    patient_labels = patient_labels[~np.isnan(patient_labels)]  # 删掉 nan
    
    patient_labels = patient_labels - 1 # 1/2/3 -> 0/1/2

    for i in range(len(patient_ids)):  # 使得所有编号长度相同
        patient_ids[i] = patient_ids[i][0:6]
    # print(dict(zip(patient_ids, patient_labels)))
    # {'sub001': 1.0}
    return dict(zip(patient_ids, patient_labels))

def _init_dataset(data_choose):
    '''
        @ return: 数据集imgs 与 标签labels，均为list类型
        从mri与excel中读取mri与标签数据
    '''
    log.logger.info("Initializing dataset {} ...".format(data_choose))
    xlsx_path = dataset_paths[data_choose]['xlsx_path']
    sheet_name = dataset_paths[data_choose]['sheet_name']
    data_path = dataset_paths[data_choose]['data_path']

    patients = _load_inf(xlsx_path, sheet_name)  # 读取病人的编号及结局（标签）
    dataset = []

    for patient_id, patient_label in patients.items():  # str, float
        log.logger.info("-- loading patients {} ...".format(patient_id))
        mri_img = _load_mri(data_path + str(patient_id) + mri_path)   # 读取dcm
        if mri_img is None: continue # 部分病人的mri数据缺失
        tumor_img = _load_tumor(data_path + str(patient_id) + tumor_path)  # 读取mha

        dataset.append({'patient_id': patient_id,
                        'patient_label':patient_label,
                        'mri_img': mri_img,
                        'tumor_img': tumor_img})

    # save as pickle
    with open(raw_data_paths[data_choose], 'wb') as f:
        pickle.dump(dataset, f, -1)
    log.logger.info("Done!")


# 生成训练集和测试集
def _generate_train_test_dataset(isCut, data_choose):
    log.logger.info("Generate train and test dataset ...")
    with open(raw_data_paths[data_choose], 'rb') as f:
        dataset = pickle.load(f)

    imgs = []
    labels = []
    for data in dataset: # for each patient
        patient_label = data['patient_label']
        mri_img = data['mri_img']
        tumor_img = data['tumor_img']

        for i in range(len(mri_img)): # for each slide
            if np.sum(tumor_img[i]) > 0: # 删除没有肿瘤区域标注的slide
                imgs.append(mri_img[i])
                labels.append(patient_label)

    dataset = {'imgs':imgs, 'labels':labels}
    
    # 分成抽样 得到训练集 和 测试集
    data_class = {'class0': [], 'class1': [], 'class2': []}
    label_class = {'class0': [], 'class1': [], 'class2': []}
    for i in range(len(dataset['labels'])):
        if dataset['labels'][i] == 0:
            data_class['class0'].append(dataset['imgs'][i])
            label_class['class0'].append(dataset['labels'][i])
        elif dataset['labels'][i] == 1:
            data_class['class1'].append(dataset['imgs'][i])
            label_class['class1'].append(dataset['labels'][i])
        else:
            data_class['class2'].append(dataset['imgs'][i])
            label_class['class2'].append(dataset['labels'][i])

    # 对两个小类别分别进行数据增广
    data_argument = transforms.Compose([
        # data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(320, padding=4),
    ])

    train_rate = 0.8 # 训练集 / 总数据集
    index = (int(len(data_class['class0']) * train_rate), 
            int(len(data_class['class1']) * train_rate), 
            int(len(data_class['class2']) * train_rate))

    train_img = data_class['class0'][:index[0]] + \
        data_class['class1'][:index[1]] + data_class['class2'][:index[2]]
    train_label = label_class['class0'][:index[0]] + \
        label_class['class1'][:index[1]] + label_class['class2'][:index[2]]

    len_train = len(train_img)
    for i in range(len_train):
        if train_label[i] == 1:
            for j in range(5):
                new_image = Image.fromarray(train_img[i], mode="I;16")
                new_image = data_argument(new_image)
                new_image = np.array(new_image)
                train_img.append(new_image)
                train_label.append(1)
        elif train_label[i] == 2:
            for j in range(12):
                new_image = Image.fromarray(train_img[i], mode="I;16")
                new_image = data_argument(new_image)
                new_image = np.array(new_image)
                train_img.append(new_image)
                train_label.append(2)

    test_img = data_class['class0'][index[0]:] + \
        data_class['class1'][index[1]:] + data_class['class2'][index[2]:]
    test_label = label_class['class0'][index[0]:] + \
        label_class['class1'][index[1]:] + label_class['class2'][index[2]:]

    # 写进pickle文件中保存
    with open(train_test_paths[data_choose], 'wb') as f:
        pickle.dump({'train_img': np.array(train_img).astype(np.float32),  # (512, 512)
                     'train_label': np.array(train_label).astype(np.long),
                     'test_img': np.array(test_img).astype(np.float32),
                     'test_label': np.array(test_label).astype(np.long)}, f, -1)

    log.logger.info("Done!")
    # return train_img, train_label, test_img, test_label

# 找到切割正方形的四个边的下标
def _find_hw():
    '''
        找到切片正方形的边长
    '''

    patients = _load_inf()  # 读取病人的编号及结局（标签）
    max_len_row = 0
    max_len_col = 0
    rx1s, rx2s, cy1s, cy2s = [], [], [], []  # 记录切割box的四个边的下标

    for p_id, label in patients.items():  # str, float
        tumor_img = _load_tumor(p_id)  # 读取mha
        if tumor_img is None:  # 部分病人的mri数据缺失
            continue

        len_col = 0  # 肿瘤列宽度
        len_row = 0  # 肿瘤行高度
        row_x1, row_x2, col_y1, col_y2 = 511, 0, 511, 0  # 每一个病人肿瘤的边界 其中x1<x2, y1<y2

        for frame in tumor_img:  # each frame is size of 512*512
            row_index_of_one = []  # 记录出现1的行的行下标

            # pass # 3.30号再继续写
            for row in range(len(frame)):  # 对于一张图片二维数组的每一层
                col_index_of_one = [i for i, x in enumerate(
                    frame[row]) if x == 1]  # 记录一行中值为1的元素的列下标
                if col_index_of_one:  # 若一行非空，即有1存在，则记录

                    if col_y1 > col_index_of_one[0]:  # 记录肿瘤的列边界点下标
                        col_y1 = col_index_of_one[0]
                    if col_y2 < col_index_of_one[-1]:
                        col_y2 = col_index_of_one[-1]

                    row_index_of_one.append(row)  # 记录出现1的行的下标

            if row_index_of_one:  # 若这一帧图片有 1
                if row_x1 > row_index_of_one[0]:  # 记录肿瘤的行边界点下标
                    row_x1 = row_index_of_one[0]
                if row_x2 < row_index_of_one[-1]:
                    row_x2 = row_index_of_one[-1]

        rx1s.append(row_x1)
        rx2s.append(row_x2)
        cy1s.append(col_y1)
        cy2s.append(col_y2)
        # 一个 病人的肿瘤长方体区域大小
        len_col = col_y2 - col_y1 + 1  # 肿瘤列宽度
        len_row = row_x2 - row_x1 + 1  # 肿瘤行高度
        if len_col > max_len_row:
            max_len_col = len_col
        if len_row > max_len_row:  # 更新最大行宽度
            max_len_row = len_row

        message = "max-min " + str((min(rx1s), max(rx2s), min(cy1s), max(cy2s))) + \
            " " + str((max_len_row, max_len_col)) + " " + \
            p_id + " " + str((row_x1, row_x2, col_y1, col_y2)) + \
            " " + str((len_row, len_col))

        log.logger.info(message)


def _cut_imgs(imgs, H=296, W=202):
    '''
        对数据集进行切片,切片后的大小 <= 295*202
    '''
    log.logger.info("Cutting images ...")
    n, c, h, w = np.shape(imgs)

    if H < 296 or W < 202:
        log.logger.info("Wrong!")
    else:
        row_x1 = 122 - int((H - 296) / 2)
        row_x2 = 418 + int((H - 296) / 2)
        col_y1 = 154 - int((H - 202) / 2)
        col_y2 = 356 + int((H - 202) / 2)

        new_imgs = np.zeros((n, c, row_x2 - row_x1, col_y2 - col_y1))

        for i in range(len(imgs)):
            new_imgs[i][0] = imgs[i][0][row_x1:row_x2, col_y1:col_y2]

    log.logger.info("Done!")
    return new_imgs


class MriDataset(Data.Dataset):
    def __init__(self, data_choose=0, train=True, isCut=False, transform=None):
        # set the paths of the images
        # assert imgs.size(0) == labels.size(0)
        # 若没有对应raw数据集的pkl文件
        if not os.path.exists(raw_data_paths[data_choose]):
            _init_dataset(data_choose)

        # 若没有对应训练测试数据集的pkl文件
        if not os.path.exists(train_test_paths[data_choose]):
            _generate_train_test_dataset(isCut, data_choose)

        with open(train_test_paths[data_choose], 'rb') as f:
            dataset = pickle.load(f)

        if train:
            self.imgs = dataset['train_img']
            self.labels = torch.from_numpy(
                dataset['train_label'].astype(np.long))
        else:
            self.imgs = dataset['test_img']
            self.labels = torch.from_numpy(
                dataset['test_label'].astype(np.long))
        self.transform = transform

    def __getitem__(self, index):
        img = self.transform(self.imgs[index])

        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)


def load_dataset(isCut=True, data_choose=0):
    '''
        @ param:
            isCut: 是否要对图片进行切割

        @ retrun: train_dataset, test_dataset
        若是第一次读取数据，则调用init_dataset函数，将加载后的数据存储到pkl中；
        若不是第一次读取数据，则读取pkl中的数据
    '''

    # mean and std of the whole dataset
    dataset_mean = [252.87963395165727]
    dataset_std = [270.5700552243061]

    # define transform operations of train dataset
    train_transform = transforms.Compose([
        # data augmentation
        #transforms.Pad(4),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=dataset_mean,
            std=dataset_std
        )])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=dataset_mean,
            std=dataset_std
        )])

    train_dataset = MriDataset(data_choose=data_choose,
                               train=True,
                               isCut=False,
                               transform=train_transform)
    test_dataset = MriDataset(data_choose=data_choose,
                              train=False,
                              isCut=False,
                              transform=test_transform)

    # compete the weights of sampler
    class_count = [0,0,0]
    for i in range(train_dataset.__len__()):
        _, target = train_dataset.__getitem__(i)
        class_count[target] += 1

    weights, targets = [], []
    class_weights = [sum(class_count) / class_count[0],
                     sum(class_count) / class_count[1], 
                     sum(class_count) / class_count[2]]

    for i in range(train_dataset.__len__()):
        _, target = train_dataset.__getitem__(i)
        weights.append(class_weights[target])

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=weights,
        num_samples=train_dataset.__len__(),
        replacement=True
    )

    return train_dataset, test_dataset, sampler


if __name__ == "__main__":
    # load_dataset(isCut=False)
    _load_mri('/home/share/Datasets/2019_rect_pcr_data/6_ROI/' +'sub001' + mri_path)
    #_load_tumor('/home/share/Datasets/2019_rect_pcr_data/6_ROI/' +
    #            'sub001' + tumor_path)

    # _load_inf(dataset_paths[2]['xlsx_path'], dataset_paths[2]['sheet_name'])
    #_init_dataset(2)
    # _generate_train_test_dataset(0)
    # load_dataset(isCut=False, data_choose=2)

    pass
