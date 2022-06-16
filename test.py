# python imports
import os
# external imports
import torch
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
# internal imports
from Model import losses
from Model.config import args
from Model.model import SpatialTransformer,  DMR
from Model.datagenerators import Atlas_Dataset, Scan_Dataset
import torch.nn.functional as F
from torch import nn


def save_image(img, ref_img, name):
    result_dir = os.path.join(args.root, args.name, args.result_dir, args.date)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(result_dir, name))


def compute_label_dice(gt, pred, dataset):
    '''The label category to be calculated, excluding background and non-existent areas in the image'''
    if dataset == 'OASIS':
        cls_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31, 32, 33, 34, 35]
    elif dataset == 'LPBA40':
        cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61,
                   62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
                   163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


# @torchsnooper.snoop()
def test(model, save, cal_per_dice, mode):
    # set up atlas tensor
    f_img = sitk.ReadImage(os.path.join(args.root, "Dataset", args.dataset, args.atlas_file))
    input_atlas = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_atlas.shape[2:]

    # Test file and anatomical labels we want to evaluate
    if mode == 'Atlas':
        test_set = Atlas_Dataset(dataset=args.dataset, file_dir=args.test_dir, root=args.root, label_dir=args.label_dir)
    elif mode == 'Scan':
        test_set = Scan_Dataset(dataset=args.dataset, file_dir=args.test_dir, root=args.root, label_dir=args.label_dir)

    if args.local_rank == 0:
        print("Number of test image pairs:{}".format(len(test_set)))
    test_loader = Data.DataLoader(dataset=test_set, batch_size=1, num_workers=1, pin_memory=True, drop_last=True)

    # Set up model
    if model == None:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(args.local_rank)
        model = DMR(len(vol_size), vol_size, layer=4)
        state_dict = torch.load(os.path.join(args.root, args.checkpoint_path))
        new_state_dict = {}
        for k in list(state_dict.keys()):
            # Copy backbone weights
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = state_dict[k]
        # print(new_state_dict.keys())
        model.load_state_dict(state_dict=new_state_dict)

        model.cuda(args.local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                    find_unused_parameters=True)

    STN_img = SpatialTransformer(vol_size).cuda(args.local_rank)
    STN_label = SpatialTransformer(vol_size, mode="nearest")
    model.eval()
    STN_img.eval()
    STN_label.eval()

    DSC = []
    Per = []
    J = []

    input_fixed = torch.from_numpy(input_atlas)
    fixed_label = \
    sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.root, "Dataset", args.dataset, args.atlas_label)))[
        np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label)

    for iter, data in enumerate(test_loader):
        # get fixed image and moving image
        if mode == 'Atlas':
            input_moving, moving_label = data

        elif mode == 'Scan':
            input_fixed, input_moving, fixed_label, moving_label = data

        input_fixed = input_fixed.cuda().float()
        input_moving = input_moving.cuda().float()
        fixed_label = fixed_label.float()
        moving_label = moving_label.float()

        # Get the image and label after registration
        pred_flow, corrs = model(input_moving, input_fixed)
        pred_img = STN_img(input_moving, pred_flow)
        pred_label = STN_label(moving_label, pred_flow.cpu().detach())

        # Obtain sub-flow of 1/2 1/4 1/8 1/16 to full resolution
        m_flow = []
        for i in range(0, 4):
            flow = F.interpolate(corrs[i], vol_size, mode='trilinear', align_corners=True)
            m_flow.append(flow)

        # Calculation of DSC, pre(J) and std(J)
        I_DSC = []
        I_J = []
        I_Per = []
        for i in range(1):
            iter_dice = compute_label_dice(fixed_label[i, ...], pred_label[i, ...], args.dataset)
            iter_J = losses.Get_Ja(pred_flow.permute(0, 2, 3, 4, 1))
            I_DSC.append(iter_dice)
            c = (iter_J < 0).nonzero()
            I_Per.append(c.size(0) / (iter_J.size(1) * iter_J.size(2) * iter_J.size(3)) * 100)
            I_J.append(iter_J.cpu().detach().numpy())
        dice = np.mean(I_DSC)
        per = np.mean(I_Per)
        j = np.std(I_J)

        if args.local_rank == 0 and cal_per_dice == True:
            print("iter:{}, dice:{}, j:{}, per:{}".format(iter, dice, j, per))
        DSC.append(dice)
        J.append(j)
        Per.append(per)

        if mode == 'Scan':
            f_img = sitk.ReadImage(os.path.join(args.root, "Dataset", args.dataset, 'test', test_set.files[iter]))

        if args.local_rank == 0 and save == True:
            moving_name = os.path.split(test_set.files[iter])[1]
            save_image(pred_img, f_img, moving_name[:4] + "_warped.nii.gz")
            save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, moving_name[:4] + "_flow.nii.gz")
            save_image(pred_label, f_img, moving_name[:4] + "_label.nii.gz")
            save_image(m_flow[0].permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, moving_name[:4] + "_inflow16.nii.gz")
            save_image(m_flow[1].permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, moving_name[:4] + "_inflow8.nii.gz")
            save_image(m_flow[2].permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, moving_name[:4] + "_inflow4.nii.gz")
            save_image(m_flow[3].permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, moving_name[:4] + "_inflow2.nii.gz")
        del pred_flow, pred_img, pred_label

    if args.local_rank == 0:
        print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC))
        print("std(J): ", np.mean(J))
        print("Per: ", np.mean(Per))
    return np.mean(DSC)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    test(model=None, save=False, cal_per_dice=True, mode='Atlas')
