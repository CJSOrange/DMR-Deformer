# python imports
import os
import warnings
import time
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch import nn
import torch.utils.data as Data
import logging
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from itertools import chain
# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators import Scan_Dataset
from Model.model import SpatialTransformer, DMR
from test import compute_label_dice, test


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs(model_dir, log_dir, result_dir, board_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(board_dir):
        os.makedirs(board_dir)


# Auxiliary loss calculation
def cal_subloss(corrs, vol, stn, input_moving, input_fixed, sim_loss_fn, grad_loss_fn, alpha):
    loss = 0
    for i in range(0, 4):
        flow = F.interpolate(corrs[i], vol, mode='trilinear', align_corners=True)*(2**(4-i))
        m2f = stn(input_moving, flow)
        sim_loss = sim_loss_fn(m2f, input_fixed)
        grad_loss = grad_loss_fn(flow)
        loss = loss + sim_loss + alpha * grad_loss
    return loss


def train():
    # Create the required folder and specify the GPU
    model_dir = os.path.join(args.root, args.name, args.model_dir, args.date)
    log_dir = os.path.join(args.root, args.name, args.log_dir, args.date)
    result_dir = os.path.join(args.root, args.name, args.result_dir, args.date)
    board_dir = os.path.join(args.root, args.name, args.board_dir, args.date)
    if args.local_rank == 0:
        make_dirs(model_dir, log_dir, result_dir, board_dir)

    # The log file
    if args.local_rank == 0:
        log_name = str(args.epoch) + "_" + str(args.lr) + "_" + str(args.alpha)
        log_file = os.path.join(log_dir, log_name)
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        logging.info("log_name:{}".format(log_name))
        logging.info("args:{}".format(args))
        writer = SummaryWriter(board_dir)

    # DDP (use 4 gpus for parallel training)
    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)

    # Read the Atlas image and gain the shape of medical image [B, C, D, W, H]
    f_img = sitk.ReadImage(os.path.join(args.root, "Dataset", args.dataset, args.atlas_file))
    input_atlas = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_atlas.shape[2:]

    # Create registration network model (4 layer DMR) and STN
    model = DMR(len(vol_size), vol_size, layer=4)
    # Load the pre-training model (checkpoint)
    if args.use_checkpoint:
        state_dict = torch.load(os.path.join(args.root, args.checkpoint_path))
        new_state_dict = {}
        for k in list(state_dict.keys()):
            # Copy backbone weights
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = state_dict[k]
        model.load_state_dict(state_dict=new_state_dict)

    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    if args.use_checkpoint:
        with torch.no_grad():
            test_Dice = test(model=model, save=True, cal_per_dice=True, mode='Atlas')
            logging.info("pretrain-Test_Dice:{}".format(test_Dice))

    stn = SpatialTransformer(vol_size)
    stn.cuda(args.local_rank)
    stn_label = SpatialTransformer(vol_size, mode="nearest")

    num_gpu = (len(args.gpu)+1) // 2
    # Number of model parameters
    logging.info("model:{} ".format(count_parameters(model)))
    logging.info("STN:{} ".format(count_parameters(stn)))

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
    # opt = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay= 1e-4, nesterov=True)

    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # Get all the names of the training data
    train_set = Scan_Dataset(dataset=args.dataset, file_dir=args.train_dir, root=args.root, label_dir=args.label_dir)
    logging.info("Number of training image pairs:{}".format(len(train_set)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = Data.DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size//num_gpu,
                                   num_workers=8, pin_memory=True, drop_last=True)

    start_time = time.time()
    torch.set_grad_enabled(True)

    # Training loop.
    for epoch in range(1, args.epoch+1):
        model.train()
        stn.train()
        train_sampler.set_epoch(epoch)  # shuffle
        start_epoch = time.time()
        DSC=[]

        for iter, data in enumerate(train_loader):
            #adjust_learning_rate(opt, epoch, args.epoch, args.lr)
            # Generate the moving images and convert them to tensors.
            input_fixed, input_moving, fixed_label, moving_label = data
            # [B, C, D, W, H]
            input_fixed = input_fixed.cuda(args.local_rank, non_blocking=True).float()
            input_moving = input_moving.cuda(args.local_rank, non_blocking=True).float()
            fixed_label = fixed_label.float()
            moving_label = moving_label.float()

            # Run the data through the model to produce warp and flow field
            flow_m2f, corrs = model(input_moving, input_fixed)
            m2f = stn(input_moving, flow_m2f)
            pred_label = stn_label(moving_label, flow_m2f.cpu().detach())

            # Calculate loss
            sim_loss = sim_loss_fn(m2f, input_fixed)
            grad_loss = grad_loss_fn(flow_m2f)
            subloss = cal_subloss(corrs, vol_size, stn, input_moving, input_fixed, sim_loss_fn, grad_loss_fn, args.alpha)
            loss = sim_loss + args.alpha * grad_loss + subloss

            with torch.no_grad():
                I_DSC = []
                for i in range(args.batch_size//num_gpu):
                    iter_dice = compute_label_dice(fixed_label[i, ...], pred_label[i, ...], args.dataset)
                    I_DSC.append(iter_dice)
                b_train_dice = np.mean(I_DSC)
                DSC.append(b_train_dice)
            if args.local_rank == 0:
                logging.info("epoch:{}  iter:{}  loss:{}  sim:{}  grad:{}  dice:{}".format(epoch, iter, loss.item(), sim_loss.item(), grad_loss.item(), b_train_dice))

            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
            '''
            if epoch % args.n_save_epoch == 0:
                # Save images
                m_name = str(epoch) + str(iter) + "_m.nii.gz"
                m2f_name = str(epoch) + str(iter) + "_m2f.nii.gz"
                test.save_image(input_moving, f_img, m_name)
                test.save_image(m2f, f_img, m2f_name)
                logging.info("warped images have saved.")
            '''

        end_epoch = time.time()

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

            if epoch % args.n_save_epoch == 0:
                # Save model checkpoint
                save_file_name = os.path.join(model_dir, '%d.pth' % epoch)
                torch.save(model.state_dict(), save_file_name)

        # test model
        with torch.no_grad():
            train_Dice = np.mean(DSC)
            test_Dice = test(model=model, save=False, cal_per_dice=False, mode='Atlas')
            logging.info("epoch:{}, Train_Dice:{}, Test_Dice:{}".format(epoch, train_Dice, test_Dice))

        # tensorboard visualization
        if args.local_rank == 0:
            writer.add_scalar('data/lr_', opt.param_groups[0]['lr'], epoch)
            writer.add_scalar('data/loss_', loss, epoch)
            writer.add_scalar('data/sim_loss_', sim_loss, epoch)
            writer.add_scalar('data/grad_loss', grad_loss, epoch)
            writer.add_scalars('data/Dice_', {'train_Dice': train_Dice, 'test_Dice': test_Dice}, epoch)
            writer.flush()

    if args.local_rank == 0:
        writer.close()

    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train()
