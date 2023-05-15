import models.UNet
from models.UNet_3Plus import *

import models2.unet_model


def count_parameters(model):
    model_param_count = 1.0 * sum(param.numel() for param in model.parameters())/1000000
    print(f"# generator parameters of Model : {model_param_count} M")

    return model_param_count


count_parameters(UNet_3Plus_DeepSup_CGM())
count_parameters(UNet_3Plus_DeepSup())
count_parameters(UNet_3Plus())
count_parameters(models.UNet.UNet())
count_parameters(models2.unet_model.UNet(n_channels=3, n_classes=1))
count_parameters(models2.unet_model.UNet(n_channels=3, n_classes=1,bilinear=False))

