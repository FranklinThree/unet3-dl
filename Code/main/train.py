import os
import time
from threading import Thread
import subprocess
import shutil
import yaml
from torch import optim, nn
import torch
import torch.utils.data
import pickle
from loss.iouLoss import IOU_loss
import models.UNet_3Plus as M
import logging
from config import yml_config
from Brain_MR_dataset import collate_fn


# is_ok = False
class TrainParam:

    def __init__(self, epochs, batch_size, lr, net, criterion, model_save_path=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.net = net
        self.criterion = criterion
        # config = None

        if model_save_path is None:
            ind = yml_config["train"]["params"]["index"]
            model_save_path = f"model-{ind}.pth"
            # 写入YAML文件
            with open("config.yml", 'w') as outfile:
                yml_config["train"]["params"]["index"] = ind + 1
                yaml.dump(yml_config, outfile, default_flow_style=False)
        self.model_save_path = model_save_path


class ThisTrainParam(TrainParam):
    def __init__(self, epochs, batch_size, lr=0.00001, deep_supervision: bool = False,
                 class_guided_module: bool = False,
                 net=M.UNet_3Plus,
                 criterion=nn.BCELoss(),
                 accumulation_steps=1,
                 from_model_path: str = None,
                 force_epoch=0):
        super().__init__(epochs, batch_size, lr, net, criterion)

        if deep_supervision:
            if class_guided_module:
                self.net = M.UNet_3Plus_DeepSup_CGM
            else:
                self.net = M.UNet_3Plus_DeepSup

        self.model = self.net(1, 1)
        # torch2.0.0加入的新特性
        # self.model = torch.compile(model, mode="default")

        device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_)
        self.accumulation_steps = accumulation_steps
        self.from_model_path = from_model_path
        # self.lr = self.lr * (accumulation_steps/2)
        self.force_epoch = force_epoch


# def train_net(net, device, epochs=40, batch_size=10, lr=0.00001, model_save_path='uname_model.pth'):
def train_net(param: ThisTrainParam):
    # 初始化tensorboardX
    # writer = SummaryWriter('result_tensorboard')
    # tensorboard_ind = 0
    logging.info("训练开始")
    logging.info("模型保存名称：%s", param.model_save_path)

    model = param.model

    # optimizer = optim.Adam(model.parameters(), lr=param.lr, betas=(0.9, 0.999))

    best_loss = float('inf')
    done_epoch = 0
    state = None
    isCon = param.from_model_path is not None
    # 如果是继续训练模式，则读取模型
    if isCon:
        logging.warning('当前处于继续训练模式，该模式存在不可预期的危险，请注意检查训练效果！')
        logging.warning("指定原模型：%s", param.from_model_path)
        state = torch.load(param.from_model_path, map_location=param.device)
        model.load_state_dict(state['model'])
        model.to(param.device)

    else:
        model.to(param.device)

    optimizer = optim.RMSprop(model.parameters(), lr=param.lr, weight_decay=1e-8, momentum=0.9)

    if isCon:
        optimizer.load_state_dict(state['optimizer'])

        # param = state['param']
        done_epoch = state['epoch']
        best_loss = state['best_loss']

    if param.force_epoch > done_epoch:
        done_epoch = param.force_epoch
    logging.info("训练基本参数：batch_size = %d, accumulation_steps = %d", param.batch_size, param.accumulation_steps)
    # 读取训练集
    with open(r'train_dataset', 'rb') as trds:
        train_dataset = pickle.load(trds)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=param.batch_size,
                                               num_workers=8,
                                               pin_memory=True,
                                               collate_fn=collate_fn,
                                               shuffle=True)

    # 定义Loss算法
    criterion = param.criterion
    # best_loss统计，初始化为正无穷
    loss = 0

    epoch = done_epoch
    # 训练epochs次
    while epoch < param.epochs:

        # 训练模式
        model.train()

        # 计算平均loss初始化
        loss_ave_p_e = 0
        loss_ave_count = 0

        # 计时每轮初始化
        t_start = time.time()

        # 初始化梯度累加
        batch_count = 0
        optimizer.zero_grad()
        # 按照batch_size开始训练
        for batch in train_loader:
            images, labels = batch
            # for images, labels in train_loader:

            # 将数据拷贝到device中
            images = torch.stack(images) \
                .to(
                device=param.device,
                non_blocking=True,
                dtype=torch.float32
            )
            labels = torch.stack(labels) \
                .to(
                device=param.device,
                non_blocking=True,
                dtype=torch.float32
            )
            # 使用网络参数，输出预测结果
            preds = model(images)
            # 计算loss
            loss = criterion(preds, labels)
            # 计算单轮平均loss
            loss_ave_p_e += loss.item()
            loss_ave_count += 1

            logging.info('loss=%f', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss

            # 损失标准化
            loss = loss / param.accumulation_steps
            # 更新参数
            loss.backward()

            # 累加到指定的 steps 后再更新参数
            if (batch_count + 1) % param.accumulation_steps == 0:
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 梯度清零
            batch_count += 1

            # optimizer.step()

            # 添加tensorboardX支持
            # writer.add_scalar('loss', loss.item(), tensorboard_ind)
            # tensorboard_ind += 1
        t_end = time.time()
        logging.info('Loss_ave/train/cost:%f/%d/%.2fs', loss_ave_p_e / loss_ave_count, epoch + 1, t_end - t_start)
        print(f'Loss_ave/train/cost:{loss_ave_p_e / loss_ave_count}/{epoch + 1}/{t_end - t_start}s')

        mid_path = os.path.join(yml_config["train"]["modelSavePath"], 'latest.pth')
        target_path = os.path.join(yml_config["train"]["modelSavePath"], param.model_save_path)



        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'param': param
        }, mid_path)
        shutil.copy(mid_path, target_path)

        with open(yml_config['train']['flagPath'], 'r', encoding='utf-8') as file:
            yml_flag = yaml.safe_load(file)
            if yml_flag['flag'] == 1:
                return

        epoch = epoch + 1


def train_net_C(param: ThisTrainParam):
    train_thread = Thread(target=train_net, args={param})
    train_thread.start()
    train_thread.join()


# def save_checkpoint(state, filename):
#     torch.save(state, filename)

def train_controller():
    while True:
        # 获取用户输入并处理
        user_input = input("Press 's' to stop training, or 'c' to continue: ")
        if user_input.upper() == "S":
            is_ok = False
            print("Training will be stopped.")
        elif user_input.upper() == "C":
            is_ok = True
            print("Training will continue.")
