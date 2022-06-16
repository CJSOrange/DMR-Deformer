import os
from shutil import copyfile
from tqdm import tqdm
import glob

'''
# The dataset directory is like:
# root
# |--Dataset
# |-- |--LPBA40
# |-- |--|-- |-- train/test/label/fixed.nii.gz/fixed_label.nii.gz
# |-- |--OASIS
# |-- |--|-- |-- data/test_data/pairs_val.csv
'''

if __name__ == "__main__":
    train_suffix = "aligned_norm.nii.gz"
    label_suffix = "aligned_seg35.nii.gz"
    # The root directory of Neurite-OASIS
    root = "root/Dataset/OASIS" # replace with your root directory for the first item root in ""

    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    label_dir = os.path.join(root, "label")
    train_source_dir = os.path.join(root, "data")
    test_source_dir = os.path.join(root, "test_data")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # create atlas for test
    atlas_file = os.path.join(train_source_dir, "OASIS_OAS1_" + str(1).zfill(4) + "_MR1", train_suffix)
    atlas_label_file = os.path.join(train_source_dir, "OASIS_OAS1_" + str(1).zfill(4) + "_MR1", label_suffix)
    copyfile(atlas_file, os.path.join(root, "fixed.nii.gz"))
    copyfile(atlas_label_file, os.path.join(root, "fixed_label.nii.gz"))

    train_files = glob.glob(os.path.join(train_source_dir, 'OASIS*'))

    # create train
    for i in tqdm(train_files):
        number = os.path.split(i)[1].split('_')[2]
        if int(number) < 438:
            source = os.path.join(i, train_suffix)
            target = os.path.join(train_dir, number + ".nii.gz")
            copyfile(source, target)

    # create test
    for i in tqdm(range(438, 458)):
        source = os.path.join(test_source_dir, "img" + str(i).zfill(4) + ".nii.gz")
        target = os.path.join(test_dir, str(i).zfill(4) + ".nii.gz")
        copyfile(source, target)

    # create label
    for i in tqdm(train_files):
        number = os.path.split(i)[1].split('_')[2]
        if int(number) < 438:
            source = os.path.join(i, label_suffix)
            target = os.path.join(label_dir, number + ".label.nii.gz")
            copyfile(source, target)

    for i in tqdm(range(438, 458)):
        source = os.path.join(test_source_dir, "seg" + str(i).zfill(4) + ".nii.gz")
        target = os.path.join(label_dir, str(i).zfill(4) + ".label.nii.gz")
        copyfile(source, target)
