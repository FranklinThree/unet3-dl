import logging
import os
import time

if __name__ == '__main__':
    base_path = r'E:\Programming\python\pytorch\Unet3+\Code\playground'
    os.chdir(base_path)
    file_path = os.path.join(base_path,r'logs\1.log')

    logging.basicConfig(filename=file_path,filemode='a',level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #
    # # 创建 Formatter 对象并设置输出格式
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #
    # # 创建 StreamHandler 对象，并将其添加到日志记录器中
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # logging.getLogger('').addHandler(stream_handler)

    logging.debug('this is a debug')