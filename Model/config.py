import argparse
import time

parser = argparse.ArgumentParser()
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# Public parameters
parser.add_argument("--name", type=str, help="experiment name",
                    dest="name", default='DMR')
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0,1,2,3')
parser.add_argument("--root", type=str, help="root_dir",
                    dest="root", default='replace with your root directory')
parser.add_argument("--dataset", type=str, help="dataset",
                    dest="dataset", default='LPBA40')
parser.add_argument("--atlas_file", type=str, help="atlas image",
                    dest="atlas_file", default='fixed.nii.gz')
parser.add_argument("--atlas_label", type=str, help="atlas label",
                    dest="atlas_label", default='fixed_label.nii.gz')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='Result')
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

# train parameters
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="train")
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=0.0004)
parser.add_argument("--epoch", type=int, help="number of iterations",
                    dest="epoch", default=2000)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='ncc')
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=1.0)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_epoch", type=int, help="frequency of model saves",
                    dest="n_save_epoch", default=100)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='Log')
parser.add_argument("--board_dir", type=str, help="board folder",
                    dest="board_dir", default='runs')
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--use_checkpoint', default=False, action='store_true',
                    dest="use_checkpoint", help='weather use checkpoint or not')

# test parameters
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='test')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='label')
parser.add_argument("--checkpoint_path", type=str, help="model weight file",
                    dest="checkpoint_path", default='Checkpoint/LPBA40')
args = parser.parse_args()