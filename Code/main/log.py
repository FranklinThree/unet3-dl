import logging
import os
from datetime import datetime
from config import yml_config


def init():
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    # config = None

    base_dir = yml_config["general"]["baseDir"]
    log_dir = os.path.join(base_dir, '../../logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_path = os.path.join(log_dir, date_string + '_' + str(yml_config['train']['params']['index']) + '.log')

    logging.basicConfig(filename=file_path, filemode='a', level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', encoding='utf-8')

    # logger_name = yml_config["general"]["name"]
    logging.info('日志记录器初始化成功')
