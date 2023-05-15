import os
import pickle

import cv2

from Brain_MR_dataset import BMR_dataset
import numpy as np
import torch
from models.UNet_3Plus import *
from models2.unet_model import *
from config import yml_config
from Brain_MR_dataset import collate_fn


def iou(y_true, y_pred, e=1e-5):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    # if union == 0:
    #     # if intersection == 0:
    #     #     return -2
    #     # else:
    #     return -1
    percent = ((intersection + e) / (union + e))
    if np.isnan(percent):
        percent = -1
    # else:
    #     if percent == 0:
    #         return -3
    return percent


def dice(y_true, y_pred, e=1e-5):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    # if union == 0:
    # if intersection == 0:
    #     return -2
    # else:
    # return -1
    percent = ((2 * intersection + e) / (union + intersection + e))
    if np.isnan(percent):
        percent = -1
    # else:
    #     if percent == 0:
    #         return -3
    return percent


def to01(array: np.array):
    array[array >= 0.5] = 1
    array[array < 0.5] = 0
    return array


def to0255(array: np.array):
    array[array >= 0.5] = 255
    array[array < 0.5] = 0
    return array


def test(model_index: int, net: nn.Module(), force_reload=False):
    # 1. 加载模型

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    # net = net
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    model_name = f"model-{model_index}.pth"
    model_path = os.path.join(yml_config["train"]["modelSavePath"], model_name)

    result_dir = os.path.join(yml_config['test']['resultBaseDir'], model_name[:-4])

    result_info_dir = os.path.join(result_dir, 'info')
    os.makedirs(result_info_dir, exist_ok=True)
    iou_arr_path = os.path.join(result_info_dir, 'iou.npy')
    dice_arr_path = os.path.join(result_info_dir, 'dice.npy')

    if not (os.path.exists(dice_arr_path) and os.path.exists(iou_arr_path)) or force_reload:
        net.load_state_dict(torch.load(model_path, map_location=device))
        # 测试模式
        net.eval()

        # 2. 加载测试集

        # 注意这里需要引入响应的dataloader才能使用
        with open('test_dataset', 'rb') as tds:
            test_dataset = pickle.load(tds)

        test_count = len(test_dataset)
        # 使用dataloader加载测试集
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  num_workers=8,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  shuffle=False)

        # 3. 测试模型

        # 初始化保存路径
        os.makedirs(result_dir, exist_ok=True)

        count = 0

        iou_arr = np.ones(test_count)
        dice_arr = np.ones(test_count)
        for images, labels in test_loader:

            # image_tensor = torch.from_numpy(image)
            images = torch.stack(images) \
                .to(device=device, dtype=torch.float32)
            preds = net(images)
            preds = preds.cpu()
            images = images.cpu()
            for i in range(preds.shape[0]):
                image_path = os.path.join(result_dir, str(count) + '_image.png')
                label_path = os.path.join(result_dir, str(count) + '_label.png')
                pred_path = os.path.join(result_dir, str(count) + '_pred.png')

                image = images[i].detach().numpy()
                pred = preds[i].detach().numpy()
                pred = to01(pred)
                label = labels[i].detach().numpy()

                n_iou = iou(label, pred)
                n_dice = dice(label, pred)
                # print('iou =', n_iou)
                # print('dice =', n_dice)
                iou_arr[count] = n_iou
                dice_arr[count] = n_dice

                image = np.expand_dims(image.squeeze(), axis=-1)
                label = np.expand_dims(to0255(label).squeeze(), axis=-1)
                pred = np.expand_dims(to0255(pred).squeeze(), axis=-1)

                cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(label_path, label, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(pred_path, pred, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                count += 1

        for i in range(len(iou_arr)):
            print("iou_array[{}] = {}".format(i, iou_arr[i]))
        iou_arr_idx = np.where(iou_arr >= 0)[0]
        iou_valid_arr = np.take(iou_arr, iou_arr_idx)
        iou_mean = np.mean(iou_valid_arr)
        np.save(iou_arr_path, iou_arr)

        for i in range(len(iou_arr)):
            print("dice_array[{}] = {}".format(i, iou_arr[i]))
        dice_arr_idx = np.where(dice_arr >= 0)[0]
        dice_valid_arr = np.take(dice_arr, dice_arr_idx)
        dice_mean = np.mean(dice_valid_arr)
        np.save(dice_arr_path, dice_arr)

        info_txt_path = os.path.join(result_info_dir, 'info.txt')
        str_iou = f"iou_mean/count:{iou_mean}/{len(iou_valid_arr)}"
        str_dice = f"dice_mean/count:{dice_mean}/{len(dice_valid_arr)}"
        with open(info_txt_path, 'w') as info_txt:
            info_txt.write(str_iou+'\n')
            info_txt.write(str_dice+'\n')
        print(str_iou)
        print(str_dice)

    else:
        dice_arr = np.load(dice_arr_path)
        iou_arr = np.load(iou_arr_path)

        for i in range(len(iou_arr)):
            print("dice_array[{}] = {}".format(i, iou_arr[i]))
        iou_arr_idx = np.where(iou_arr >= 0)[0]
        iou_valid_arr = np.take(iou_arr, iou_arr_idx)
        iou_mean = np.mean(iou_valid_arr)

        for i in range(len(iou_arr)):
            print("iou_array[{}] = {}".format(i, iou_arr[i]))
        dice_arr_idx = np.where(dice_arr >= 0)[0]
        dice_valid_arr = np.take(dice_arr, dice_arr_idx)
        dice_mean = np.mean(dice_valid_arr)

        print(f"iou_mean/count:{iou_mean}/{len(iou_valid_arr)}")
        print(f"dice_mean/count:{dice_mean}/{len(dice_valid_arr)}")


if __name__ == '__main__':
    test(91, UNet(1, 1),
          # force_test=True
         )
    # test(82, UNet(1, 1),
    #       # force_test=True
    #      )
    # test(60, UNet_3Plus(1, 1),
    #      # force_test=True
    #      )
