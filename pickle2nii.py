import pickle
import nibabel as nib
import numpy as np

# 假设pkl文件路径为 'merged_modalities.pkl'
pkl_file_path = '/mnt/K/WHZ/datasets/BraTS2020 T+V/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_299/BraTS20_Training_299_data_f32b0_2.pkl'
nii_file_path = '/mnt/K/WHZ/datasets/merged_modalities.nii'


# 加载PKL文件
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

# 获取images数据
images = data[0]  # images是第一个元素

# 检查数据类型和形状
print(f"Type of images: {type(images)}")
print(f"Shape of images: {np.shape(images)}")

# 确保数据是numpy数组
images = np.array(images, dtype='float32')

# 检查数据形状是否是规则的；四个模态images.shape[-1] == 4，两个模态images.shape[-1] == 2
if len(images.shape) == 4 and images.shape[-1] == 2:
    # 创建NIfTI图像对象
    nii_image = nib.Nifti1Image(images, affine=np.eye(4))

    # 保存NIfTI文件
    nib.save(nii_image, nii_file_path)
    print(f"Saved NIfTI file to {nii_file_path}")
else:
    print(f"Unexpected data shape: {images.shape}. Expected shape (240, 240, 155, 4)")


