import models.UNet_3Plus
import models2.unet_model
import log
import logging
import train
from Brain_MR_dataset import BMR_dataset
from MRI_Dataset import MRI_Loader
import torch.nn
import config

if __name__ == '__main__':
    log.init()
    logging.info("进入主函数")
    param = train.ThisTrainParam(epochs=1000, batch_size=4, lr=0.00001,
                                 net=models2.unet_model.UNet,
                                 # net=models.UNet_3Plus.UNet_3Plus,
                                 criterion=torch.nn.BCEWithLogitsLoss(),
                                 from_model_path=config.getFullModelPath(10018)
                                 )
    train.train_net_C(param)