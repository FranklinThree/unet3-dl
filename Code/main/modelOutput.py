import models2.unet_model
import torch
# from UNet_3Plus import UNet_3Plus

if __name__ == '__main__':
    model = models2.unet_model.UNet(n_channels=3, n_classes=1,bilinear=False) # 实例化模型
    input_shape = (3, 256, 256)
    input_names = ['input']
    output_names = ['output']
    path = 'models/else/Unet3.onnx'
    torch.onnx.export(model, torch.randn(1, *input_shape), path, verbose=True, input_names=input_names, output_names=output_names)