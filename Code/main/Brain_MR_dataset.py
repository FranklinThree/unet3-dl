import logging
import pickle
import cv2
import os
import glob

import torch
from torch.utils.data import Dataset
import random
import config


class BMR_dataset(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label').replace('.png', '_label.png')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        # print("image.shape", image.shape)
        # print("label.shape", label.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # print("image.shape", image.shape)
        # print("label.shape", label.shape)

        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        # print("image.shape", image.shape)
        # print("label.shape", label.shape)

        # print("_____________________________")
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 12, 13, 14, 15, 16, 17])
        if flipCode < 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

def collate_fn(batch):
    images = [torch.as_tensor(item[0], dtype=torch.float32) for item in batch]
    labels = [torch.as_tensor(item[1], dtype=torch.float32) for item in batch]
    return images, labels

if __name__ == '__main__':
    train_dataset = BMR_dataset(config.yml_config["train"]["datasetPath"])

    test_dataset = BMR_dataset(config.yml_config["test"]["datasetPath"])
    # print(test_dataset.__len__())
    test_dataset.__getitem__(0)
    # 保存测试集和训练集到文件
    trds = open(os.path.join(config.yml_config['general']['baseDir'], 'train_dataset'), 'wb')
    pickle.dump(train_dataset, trds, -1)
    logging.info("训练集数量",train_dataset.__len__())
    trds.close()

    tds = open(os.path.join(config.yml_config['general']['baseDir'], 'test_dataset'), 'wb')
    pickle.dump(test_dataset, tds, -1)
    logging.info("测试集数量",test_dataset.__len__())
    tds.close()

