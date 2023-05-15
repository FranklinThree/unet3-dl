import SimpleITK as sitk
import os
import numpy as np


def mha_to_png_slices(mha_input_file, source_filename,output_directory, isLabel = False):
    label_a = ''
    if isLabel:
        label_a = 'label_'

    # 读取MHA文件
    image = sitk.ReadImage(mha_input_file)

    # 将图像转换为numpy数组
    array = sitk.GetArrayFromImage(image)

    # 获取z轴上的切片数量
    num_slices = array.shape[0]

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    max = 0
    # 遍历每个切片并将其保存为PNG文件
    for i in range(num_slices):
        slice_array = array[i]
        max_ = np.max(slice_array)
        if max_ > max :
            max = max_

        output_path = os.path.join(output_directory, source_filename[:-4] + '_slice_' + label_a + f"{i:04d}.png")

        # 如果数组只包含0和1的short值，则将其转换为二值图像（uint8）
        if np.min(slice_array) == 0 and np.max(
                slice_array) == 1:
            slice_array = (slice_array * 255).astype(np.uint8)
            slice_image = sitk.GetImageFromArray(slice_array)

            sitk.WriteImage(slice_image, output_path)
            # print(1)
        # elif np.max(slice_array) <= 255:
        #     slice_array = slice_array.astype(np.uint8)
            # print(8)
        elif np.max(slice_array) <= 65535:
            slice_image = sitk.GetImageFromArray(slice_array)
            img_8bit = sitk.RescaleIntensity(slice_image)
            img_8bit = sitk.Cast(img_8bit, sitk.sitkUInt8)
            sitk.WriteImage(img_8bit, output_path)
            # slice_array = (((slice_array / (65536 / 16) ) * 256)).astype(np.uint8)

            # slice_array = slice_array.astype(np.uint16)
            # print(16)

        # slice_array.astype(np.uint8)
        # print(np.shape(slice_array),np.max(slice_array),np.min(slice_array))


        # np.save(output_path[:-4]+".npy",{'pic':slice_image,'arr':np.ndarray,'allow_pickle':True})

    # print(max)

def slice_(source_directory, output_directory, isLabel = False):
    # 创建输出文件夹（若不存在）
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历输入文件夹下所有的MHA文件
    for filename in os.listdir(source_directory):
        if filename.endswith('.mha'):
            mha_to_png_slices(os.path.join(source_directory, filename), filename, output_directory, isLabel)
        # break


if __name__ == '__main__':
    source_directory_label = r"E:\Programming\python\pytorch\Unet3+\Datasets\Brain_MR_Angiography\TubeTK\original\Vessel_MHA"
    output_directory_label = r"E:\Programming\python\pytorch\Unet3+\Datasets\Brain_MR_Angiography\TubeTK\MRA_done\label"
    #
    slice_(source_directory_label, output_directory_label, isLabel=True)
    #
    source_directory_image = r"E:\Programming\python\pytorch\Unet3+\Datasets\Brain_MR_Angiography\TubeTK\original\MRA"
    output_directory_image = r"E:\Programming\python\pytorch\Unet3+\Datasets\Brain_MR_Angiography\TubeTK\MRA_done\image"
    #
    #
    slice_(source_directory_image, output_directory_image,isLabel=False)

    # for filename in os.listdir(output_directory_image):
    #
    #     # image = sitk.ReadImage(os.path.join(output_directory_image,filename))
    #     # array = sitk.GetArrayFromImage(image)
    #     if filename.endswith(".npy"):
    #         image = np.load(os.path.join(output_directory_image,filename),allow_pickle=True)
    #         print(filename)
    #         # break




