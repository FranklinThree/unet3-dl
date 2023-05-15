import os.path

import yaml

# config = None
# 打开yaml文件并读取内容
with open(r'config.yml', 'r', encoding='utf-8') as file:
    yml_config = yaml.safe_load(file)


def getFullModelPath(index: int):
    model_name = f"model-{index}.pth"
    model_save_path = yml_config['train']['modelSavePath']
    return os.path.join(model_save_path, model_name)
