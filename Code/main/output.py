import torch
import models.UNet_3Plus as M
if __name__ == '__main__':
    model = M.UNet_3Plus() # 实例化模型
    input_shape = (3, 256, 256)
    input_names = ['input']
    output_names = ['output']
    path = 'Unet3_plus.onnx'
    torch.onnx.export(model, torch.randn(1, *input_shape), path, verbose=True, input_names=input_names, output_names=output_names)