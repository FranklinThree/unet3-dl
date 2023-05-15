import os.path
import shutil
import re
import random

def manage_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
def divide_dataset(dataset_source_path, train_dataset_target_path, test_dataset_target_path, train_ratio = 0.8):

    image_s_dir = os.path.join(dataset_source_path, 'image')
    label_s_dir = os.path.join(dataset_source_path, 'label')

    image_test_dir = os.path.join(test_dataset_target_path, 'image')
    label_test_dir = os.path.join(test_dataset_target_path, 'label')

    image_train_dir = os.path.join(train_dataset_target_path, 'image')
    label_train_dir = os.path.join(train_dataset_target_path, 'label')

    manage_dir(image_s_dir)
    manage_dir(label_s_dir)

    manage_dir(image_test_dir)
    manage_dir(label_test_dir)

    manage_dir(image_train_dir)
    manage_dir(label_train_dir)

    filename_list = os.listdir(image_s_dir)
    full_amount = len(filename_list)
    print('instances count:', full_amount)
    train_amount = int(full_amount*train_ratio)
    test_amount = full_amount-train_amount
    print('train instances count:', train_amount)
    print('test instances count:', full_amount-train_amount)

    filename_train_list = random.sample(filename_list,train_amount)
    filename_test_list = [x for x in filename_list if x not in filename_train_list]

    for filename in filename_train_list:
        # 简化重置图片名称
        filename_suffix = filename[-4:]
        filename_n = re.sub(r'\D', '', filename).strip()
        filename_n = filename_n[:3]+'_'+filename_n[3:]+filename_suffix

        # 复制image
        image_source_full_path = os.path.join(image_s_dir, filename)
        shutil.copy(image_source_full_path,os.path.join(train_dataset_target_path,'image', filename_n))

        # 复制label
        label_s_filename = filename[:10]+'VascularNetwork_slice_label_'+filename[-8:]
        label_source_full_path = os.path.join(label_s_dir, label_s_filename)
        shutil.copy(label_source_full_path,
                    os.path.join(train_dataset_target_path,'label', filename_n[:-4]+'_label'+filename_n[-4:]))
        # break

    for filename in filename_test_list:
        # 简化重置图片名称
        filename_suffix = filename[-4:]
        filename_n = re.sub(r'\D', '', filename).strip()
        filename_n = filename_n[:3] + '_' + filename_n[3:] + filename_suffix

        # 复制image
        image_source_full_path = os.path.join(image_s_dir, filename)
        shutil.copy(image_source_full_path, os.path.join(test_dataset_target_path, 'image', filename_n))

        # 复制label
        label_s_filename = filename[:10] + 'VascularNetwork_slice_label_' + filename[-8:]
        label_source_full_path = os.path.join(label_s_dir, label_s_filename)
        shutil.copy(label_source_full_path,
                    os.path.join(test_dataset_target_path, 'label', filename_n[:-4] + '_label' + filename_n[-4:]))
        # break





if __name__ == '__main__':
    base_dataset_path = r'/Datasets/Brain_MR_Angiography/TubeTK'
    dataset_source_path = os.path.join(base_dataset_path, r'MRA_done')
    train_dataset_target_path = os.path.join(base_dataset_path, r'train')
    test_dataset_target_path = os.path.join(base_dataset_path, r'test')
    divide_dataset(dataset_source_path,train_dataset_target_path,test_dataset_target_path)