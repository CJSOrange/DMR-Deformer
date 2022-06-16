import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
import random
from scipy import ndimage

'''
# The dataset directory is like:
# root
# |--Dataset
# |-- |--LPBA40
# |-- |--|-- |-- train/test/label/fixed.nii.gz/fixed_label.nii.gz
# |-- |--OASIS
# |-- |--|-- |-- data/test_data/pairs_val.csv
# |-- |--|-- |-- train/test/label/fixed.nii.gz/fixed_label.nii.gz

Two datasets: 
Scan_Dataset for training/ Scan-to-Scan
Atlas_Dataset for testing/ Scan-to-Atlas
    :param dataset: LPBA40 or OASIS.
    :param file_dir: train or test.
    :param root: your root directory of the Dataset
    :param label_dir: label directory.    
'''


class Scan_Dataset(Data.Dataset):
    def __init__(self, dataset, file_dir, root, label_dir):
        # initialization
        self.dataset_file = os.path.join(root, "Dataset", dataset)
        self.files = sorted(glob.glob(os.path.join(self.dataset_file, file_dir, '*.nii.gz')))
        self.root = root
        self.label_dir = label_dir

    def __len__(self):
        # Returns the size of the dataset
        return len(self.files)-1

    def __getitem__(self, index):
        # Index a certain data in the data set, you can also preprocess the data
        fixed_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        moving_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index+1]))[np.newaxis, ...]

        fixed_name = os.path.split(self.files[index])[1]
        moving_name = os.path.split(self.files[index+1])[1]
        fixed_label_file = glob.glob(os.path.join(self.dataset_file, self.label_dir, fixed_name[:4] + "*"))[0]
        fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_file))[np.newaxis, ...]
        moving_label_file = glob.glob(os.path.join(self.dataset_file, self.label_dir, moving_name[:4] + "*"))[0]
        moving_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_file))[np.newaxis, ...]

        # The return value is automatically converted to torch's tensor type
        return fixed_img_arr, moving_img_arr, fixed_label, moving_label


class Atlas_Dataset(Data.Dataset):
    def __init__(self, dataset, file_dir, root, label_dir):
        self.dataset_file = os.path.join(root, "Dataset", dataset)
        self.files = sorted(glob.glob(os.path.join(self.dataset_file, file_dir, '*.nii.gz')))
        self.root = root
        self.label_dir = label_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        moving_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        moving_name = os.path.split(self.files[index])[1]
        moving_label_file = glob.glob(os.path.join(self.dataset_file, self.label_dir, moving_name[:4] + "*"))[0]
        moving_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_file))[np.newaxis, ...]
        return moving_img_arr, moving_label


def Random_Flip(image1, image2, label1, label2):
    if random.random() < 0.5:
        image1 = np.flip(image1, 1)
        label1 = np.flip(label1, 1)
        image2 = np.flip(image2, 1)
        label2 = np.flip(label2, 1)
    if random.random() < 0.5:
        image1 = np.flip(image1, 2)
        label1 = np.flip(label1, 2)
        image2 = np.flip(image2, 2)
        label2 = np.flip(label2, 2)
    if random.random() < 0.5:
        image1 = np.flip(image1, 3)
        label1 = np.flip(label1, 3)
        image2 = np.flip(image2, 3)
        label2 = np.flip(label2, 3)

    return image1.copy(), image2.copy(), label1.copy(), label2.copy()


def Random_intencity_shift(image1, image2, label1, label2, factor=0.1):
    scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, 1, image1.shape[-2], image1.shape[-1]])
    shift_factor = np.random.uniform(-factor, factor, size=[1, 1, image1.shape[-2], image1.shape[-1]])

    image1 = image1*scale_factor+shift_factor
    image2 = image2 * scale_factor + shift_factor
    return image1, image2, label1, label2


def Random_rotate(image1, image2, label1, label2):
    angle = round(np.random.uniform(-10, 10), 2)
    image1 = ndimage.rotate(image1, angle, axes=(1, 2), reshape=False)
    label1 = ndimage.rotate(label1, angle, axes=(1, 2), reshape=False)
    image2 = ndimage.rotate(image2, angle, axes=(1, 2), reshape=False)
    label2 = ndimage.rotate(label2, angle, axes=(1, 2), reshape=False)
    return image1, image2, label1, label2


def transform(image1, image2, label1, label2):
    image1, image2, label1, label2 = Random_Flip(image1, image2, label1, label2)
    #image1, image2, label1, label2 = Random_rotate(image1, image2, label1, label2)
    #image1, image2, label1, label2 = Random_intencity_shift(image1, image2, label1, label2)
    return image1, image2, label1, label2


if __name__ == "__main__":
    root="/apdcephfs/share_1290796/jiashunchen/ft_local/Reg"
    DS = Atlas_Dataset('LPBA40', 'test', root, 'label')
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    for epoch in range(1):
        for iter, data in enumerate(DL):
            # Generate the moving images and convert them to tensors.
            #fixed_img_arr, moving_img_arr, fixed_label, moving_label = data
            moving_img_arr, moving_label = data
            print("{}:epoch,{}:iter,{}:datasize {}:labelsize".format(epoch, iter, moving_img_arr.size(), moving_label.size()))