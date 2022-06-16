# DMR

Official PyTorch implementation for MICCAI 2022 paper, Deformer: Towards Displacement Field Learning for Unsupervised Medical Image Registration.

## Directory structure

The overall code directory structure is as follows:

------

- `Checkpoint`：Store the pre-trained model pth for LPBA40 and Neurite-OASIS；
- `Model`
  - `config.py`：Model configuration file, used to specify learning rate, training times, loss weight, batch size, save interval, etc., is divided into three parts, namely public parameters, training parameters and test parameters.
  - `configs.py`：VIT model configuration file.
  - `datagenerators.py`：The torch.utils.data package is used to provide data according to the path of the image file.
  - `losses.py`：Calculation of various loss functions, including smoothness loss of deformation field, MSE, DSC, NCC, CC, the number of negative values in jacobian determinant, etc.
  - `model.py`：registration network (DMR) and space transformation network (STN), and the three are modular separation.
  - `Deformable_skip_learner.py`：Implementation of the refining network.
  - `Deformer.py`：Implementation of Deformer, you can modify the number of displacement base and heads.
  - `ResNet3D.py`:  3D Resnet for ablation study.
  - `VIT.py`: VIT for ablation study.
- `test.py`：code for testing.
- `train.py`:  code for training.
- `preprocess.py`:  preprocess for Neurite-OASIS, split into train set and test set.

------

## Create the environment with Anaconda

```
$ conda create -n DMR python=3.6
$ source activate DMR
$ conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.1 -c pytorch
$ conda install scikit-learn=0.24.2
$ pip install tensorboardX
$ conda install matplotlib
$ pip install SimpleITK
$ pip install tqdm
```

## Prepare datasets

You can download the pre-processed datasets [LPBA40](https://drive.google.com/file/d/1308rPiQBZTa13tI-0KbGYUv41G88ejjf/view?usp=sharing) and  [Neurite-OASIS](https://drive.google.com/file/d/1VmwQs2nCsRHEHKUtRUAIE-DJqX6XD4iq/view?usp=sharing) from Google Driver. 

For Neurite-OASIS，you can also alternative download Full Dataset，Validation (skull stripped) and pairs_val.csv from [learn2reg2021](https://learn2reg.grand-challenge.org/Learn2Reg2021/) task3 official website. Then put these three directories under 'root/Dataset/OASIS', where root is optional. Finally, rename Full Dataset as data, Validation (skull stripped) as test_data. Change the root parameter in propress.py and run it.

The final dataset directory structure is as follows:

- `Root`
  - Dataset
    - LPBA40
      - train/test/label/fixed.nii.gz/fixed_label.nii.gz
    - OASIS
      - data/test_data/pairs_val.csv
      - train/test/label/fixed.nii.gz/fixed_label.nii.gz

## Pre-trained model

You can download the pre-trained DMR model [LPBA40.pth](https://drive.google.com/file/d/1JRALMQvJXybCQLdi9eRvmHw6ZQ9f95IX/view?usp=sharing) and [Neurite-OASIS.pth](https://drive.google.com/file/d/1VYwsqmAbYFTVbZS8mTlbX6r852uqbvGK/view?usp=sharing) from Google driver.

Then put this two .pth file under 'Checkpoint' directory.

## Command to run DMR

First, you should modify the root  parameter in config.py and replace with your root directory (The root directory should include dataset and experiment results). Note that our default setting is DDP distributed training on 4 GPUs.

You can run DMR on LPBA40 with 

```
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py --name='experiment_name' --lr=0.0004 --batch_size=4 --dataset='LPBA40' --gpu='0,1,2,3'
```

For Neurite-OASIS, you can use 

```
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py --name='other_experiment_name' --lr=0.001 --batch_size=4 --dataset='OASIS' --gpu='0,1,2,3'
```

If you want to load checkpoint at the beginning of training step, you can run this command,  note that ‘path’ is the relative path to root. You can also see the test results for the DMR model loaded with this checkpoint.

```
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py --name='experiment_name' --lr=0.001 --batch_size=4 --dataset='LPBA40' --gpu='0,1,2,3' --use_checkpoint --checkpoint_path='path'
```

## Results

After training and testing, you can see all results in the  directory 'root/experiment'  as following:

- `root`
  - `Experiment`：The experiment named by you, there can be many experiment names for each dataset.
    - `Checkpoint`：store trained models.
    - `Log`：store log files to record changes in loss values for each parameter.
    - `Result`：store image data generated during training and testing.
    - `run`：Store event files, prepare loss and learning rate for Tensorboard visualization.