# coding: utf-8
'''
    2019/3/29

    读取6_ROI中的347个病人的mri数据：sub1 ~ sub364
    数据集：(肿瘤医院，中山六院)
        T2: 512*512*N (N为帧数，不同病人的MRi帧数不同)
'''

import logging
import os
import pickle
from logging import handlers

import matplotlib.pylab as plt
import numpy as np
import pydicom
import SimpleITK as sitk

import xlrd

# the path of the information xlsx
inf_xlsx_path = "../data/information.xlsx" 
# the path of datasets in the hospital of the patients
dataset_dir = "../data/6_ROI/" 
# the sub-path of mri image for each patient
mri_path = "/MRI/T2" 
# the sub-path of tumor mri image for each patient
tumor_path = "/MRI/T2tumor.mha"
# mri the final path is: dataset_dir + “sub001” + mri_path
# mha the final path is: dataset_dir + “sub001” + tumor_path

# the path of pkl of dataset
save_file = dataset_dir + "mri_dataset.pkl"
# the path of log for dabugging and report
log_path = './../logs/all.log'

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backCount, encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


def _load_mri(p_id):
    '''
        @ param: p_id为str类型，表示病人的编号
        @ return: 返回这个病人的mri图像的列表(N,512,512) 

        读取病人的T2的mri图像，图像文件格式为dcm
    '''
    img_path = dataset_dir + p_id + mri_path
    image_array = None
    # use SimpleITK to load dcm file
    try:    
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(img_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        origin = image.GetOrigin()  # x, y, z
        spacing = image.GetSpacing()  # x, y, z
    except:
        print('Cannot find file \"' + img_path + '\"')
    return image_array


def _load_tumor(p_id):
    '''
        @ param: p_id为str类型，表示病人的编号
        @ return: 返回这个病人的tumor标注的mri图像的(N,512,512)

        读取病人的T2的mri图像的肿瘤标记图像，图像文件格式为mha

        mha为三维数组，元素为0/1 int
    '''
    img_path = dataset_dir + p_id + tumor_path
    image_array = None
    try:
        image = sitk.ReadImage(img_path)
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        origin = image.GetOrigin()  # x, y, z
        spacing = image.GetSpacing()  # x, y, z
    except:
        print('Cannot find file \"' + img_path + '\"')
    return image_array

def _load_inf():
    ''' 
        @ return: dict类型 {p_id: label\}
        读取information.xlsx文件，加载病人编号，标签
        
        for 6_ROI:
        index in excel: 0 p_id, 65 label
        
        先读取6_ROI
        # sub338（术后4周肝转移），手动在excel去掉括号及括号中的内容
    '''
    # patients = {} # {p_id:label }
    excel_file = xlrd.open_workbook(inf_xlsx_path) # 打开excel文件
    sheet_names = excel_file.sheet_names()  # ['CC_ROI', '6_ROI']

    sheet_6_ROI = excel_file.sheet_by_index(1) # '6_ROI'
    # 下面这两行代码只对于6_ROI
    p_ids = sheet_6_ROI.col_values(0)[1:348] # 去掉title, 简单的去掉末尾的空字符的元素
    labels = sheet_6_ROI.col_values(65)[1:348]

    # print(dict(zip(p_ids, labels)))
    return dict(zip(p_ids, labels))


def _init_dataset():
    '''
        @ return: 数据集imgs 与 标签labels，均为list类型
        从mri与excel中读取mri与标签数据
    '''
    patients = _load_inf() # 读取病人的编号及结局（标签）
    imgs = []
    labels = []
    for p_id, label in patients.items(): # str, float  
        print("-- loading patients",p_id,"...")
        mri_img = _load_mri(p_id)   # 读取dcm
        if mri_img is None: # 部分病人的mri数据缺失
            continue
        # mha_img = _load_tumor(p_id)  # 读取mha
    
        for frame in mri_img:
            imgs.append([frame])
            labels.append(label-1) # excel中为1，2，3，这里变成0，1，2
        
        # break
    return imgs, labels


# 找到切割正方形的四个边的下标
def _find_hw():
    '''
        找到切片正方形的边长
    '''
    log = Logger(log_path, level='debug')

    patients = _load_inf()  # 读取病人的编号及结局（标签）
    max_len_row = 0
    max_len_col = 0
    rx1s, rx2s, cy1s, cy2s = [], [], [], [] # 记录切割box的四个边的下标

    for p_id, label in patients.items():  # str, float
        mha_img = _load_tumor(p_id)  # 读取mha
        if mha_img is None:  # 部分病人的mri数据缺失
            continue

        len_col = 0 # 肿瘤列宽度
        len_row = 0 # 肿瘤行高度
        row_x1, row_x2, col_y1, col_y2 = 511,0,511,0 # 每一个病人肿瘤的边界 其中x1<x2, y1<y2

        for frame in mha_img: # each frame is size of 512*512
            row_index_of_one = [] # 记录出现1的行的行下标

            # pass # 3.30号再继续写
            for row in range(len(frame)): # 对于一张图片二维数组的每一层
                col_index_of_one = [i for i, x in enumerate(
                    frame[row]) if x == 1]  # 记录一行中值为1的元素的列下标
                if col_index_of_one:  # 若一行非空，即有1存在，则记录

                    if col_y1 > col_index_of_one[0]: # 记录肿瘤的列边界点下标
                        col_y1 = col_index_of_one[0]
                    if col_y2 < col_index_of_one[-1]:
                        col_y2 = col_index_of_one[-1]

                    row_index_of_one.append(row) # 记录出现1的行的下标
            
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
    print("Cutting images ...")
    n, c, h, w = np.shape(imgs)

    if H < 296 or W < 202:
        print("Wrong!")
    else:
        row_x1 = 122 - int((H - 296) / 2)
        row_x2 = 418 + int((H - 296) / 2)
        col_y1 = 154 - int((H - 202) / 2)
        col_y2 = 356 + int((H - 202) / 2)
        
        new_imgs = np.zeros((n, c, row_x2-row_x1, col_y2-col_y1))
        
        for i in range(len(imgs)):
            new_imgs[i][0] = imgs[i][0][row_x1:row_x2, col_y1:col_y2]
            
    print("Done!")
    return new_imgs
    
def load_dataset():
    '''
        @ retrun: train_data,train_label,test_data,test_label
        若是第一次读取数据，则调用init_dataset函数，将加载后的数据存储到pkl中；
        若不是第一次读取数据，则读取pkl中的数据
    '''
    if not os.path.exists(save_file): # 若没有pkl文件
        print("Initializing dataset ...")
        imgs, labels = _init_dataset() # 初始化数据集
        print("Done!")
        dataset = {"imgs":imgs, 'labels':labels} # 将数据类型转化为字典类型
        print("Creating pickle file ...")
        with open(save_file, 'wb') as f: # 将数据写入pkl文件中
            pickle.dump(dataset, f, -1)
        print("Done!")
    else:
        # load mri and label data from pkl file
        with open(save_file, 'rb') as f:  
            print("Loading mridata.pkl ...")
            dataset = pickle.load(f)
            print("Done!")
    
    dataset['imgs'] = np.array(dataset['imgs'])
    dataset['labels'] = np.array(dataset['labels'])

    # 打印样本总数，每种样本的数量等信息
    print("Number of sample:", np.shape(dataset['imgs'])[0])
    print("The number of Class 1 sample:{} ({})".format(np.sum(dataset['labels'] == 0),
                                                        np.sum(dataset['labels'] == 0) / np.shape(dataset['imgs'])[0]))
    print("The number of Class 2 sample:{} ({})".format(np.sum(dataset['labels'] == 1),
                                                        np.sum(dataset['labels'] == 1) / np.shape(dataset['imgs'])[0]))
    print("The number of Class 3 sample:{} ({})".format(np.sum(dataset['labels'] == 2),
                                                        np.sum(dataset['labels'] == 2) / np.shape(dataset['imgs'])[0]))

    # 切割肿瘤区域
    dataset['imgs'] = _cut_imgs(dataset['imgs'], 300, 300)
    
    # 训练集与测试集的比例划分 ？？？？ 3:2, 5:4, ...
    rate = 0.6
    index = int(len(dataset)*rate)

    train_img = dataset['imgs'][:index]
    train_label = dataset['labels'][:index]

    test_img = dataset['imgs'][index:]
    test_label = dataset['labels'][index:]

    return train_img, train_label, test_img, test_label


if __name__ == "__main__":
    load_dataset()

    # log = Logger(log_path, level='debug')

    pass
