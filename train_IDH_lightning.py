import argparse
import os

import time

import torch.backends.cudnn as cudnn
import torch.optim

import lightningModel

import pytorch_lightning as pl
import wandb
import random

import lightningModelContrast

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='yanzhi', type=str)

parser.add_argument('--experiment', default='TransBraTS_IDH', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBraTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='/mnt/K/WHZ/datasets/BraTS2020T+V/', type=str)

parser.add_argument('--train_dir', default='BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', type=str)

parser.add_argument('--valid_dir', default='BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData', type=str)

parser.add_argument('--test_dir', default='BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train_label_add.txt', type=str)  # IDH_all.txt

parser.add_argument('--valid_file', default='valid_label.txt', type=str)  # IDH_test.txt

parser.add_argument('--test_file', default='test_label.txt', type=str)  # IDH_test.txt

parser.add_argument('--dataset', default='brats_IDH', type=str)

parser.add_argument('--model_name', default='TransBraTS', type=str)

parser.add_argument('--input_C', default=4, type=int)

parser.add_argument('--input_H', default=240, type=int)

parser.add_argument('--input_W', default=240, type=int)

parser.add_argument('--input_D', default=155, type=int)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=0.0001, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice',
                    type=str)  # softmax_dice常见于医学影像分割任务中，它是一种结合了Softmax交叉熵损失和Dice损失的组合损失函数。

parser.add_argument('--num_class', default=6, type=int)  # 4

parser.add_argument('--seed', default=1234, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=0, type=int)  # 8

# parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--start_epoch', default=1, type=int)

parser.add_argument('--end_epoch', default=1000, type=int)

parser.add_argument('--save_freq', default=750, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

#下面时涉及到WSI的参数
parser.add_argument('--cancer',
                         type=str, default='ALL',
                         help='Cancer type')
parser.add_argument('--backbone',
                         type=str, default='resnet18',  # swin_transformer
                         help='backbone model, for example, resnet-50, mobilenet, etc')
parser.add_argument('--outcome',
                         type=str, default='idh,atrx,p19q',
                         help='name of the outcome variable')
parser.add_argument('--num-classes',
                         type=int, default=2,
                         help='number of outputs of the model, only used for classification')
parser.add_argument('--magnification',
                         type=str, default='20,',
                         help='magnification level')

# specify the path of the meta files
parser.add_argument('--val-meta',
                         type=str, default='./dataset/',
                         help='path to the meta file for the evaluation portion')
parser.add_argument('--train-meta',
                         type=str, default='./dataset/',
                         help='path to the meta file for the training portion')
parser.add_argument('--patch-meta',
                         type=str, default='./dataset/patches_meta',
                         help='path to the meta file for the training portion')

# specify patch manipulations
parser.add_argument('--crop-size',
                         type=int, default=256,
                         help='size of the crop')
parser.add_argument('--num_patches',
                         type=int, default=64,
                         help='number of patches to select from one patient during one iteration')
# specify experiment details
# parser.add_argument('-m', '--mode',
#                          type=str, default='train',
#                          help='mode, train or test')
parser.add_argument('--patience',
                         type=int, default=1000,
                         help='break the training after how number of epochs of no improvement')
parser.add_argument('--epochs',
                         type=int, default=1000,
                         help='total number of epochs to train the model')
parser.add_argument('--pretrain',
                         action='store_true', default=True,
                         help='whether use a pretrained backbone')
parser.add_argument('--random-seed',
                         type=int, default=88,
                         help='random seed of the model')

# data specific arguments
parser.add_argument('-b', '--batch_size',
                         type=int, default=1,
                         help='batch size')  # 最终实际的batch_size为args.batch_size * args.num_patches
parser.add_argument('-vb', '--vbatch-size',
                         type=int, default=1,
                         help='val batch size')
parser.add_argument('--repeats_per_epoch',
                         type=int, default=1,
                         help='how many times to select one patient during each iteration')
# parser.add_argument('--num-workers',
#                          type=int, default=0,
#                          help='number of CPU threads')
# evaluation details
parser.add_argument('--sample-id',
                         action='store_true', default=False,
                         help='if true, sample patches by patient; otherwise evaluate the model on all patches')
parser.add_argument('--num-val',
                         type=int, default=64,
                         help='number of patches to select from one patient during validation')
parser.add_argument('--num-crops',
                         type=int, default=1,
                         help='新加的')

args = parser.parse_args()


def main_worker():
    # model = lightningModel.LightningModel(args)
    model = lightningModelContrast.LightningModel(args)

    trainer = pl.Trainer(
        strategy='ddp_find_unused_parameters_true',
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        benchmark=True,
        max_epochs=args.end_epoch,
        check_val_every_n_epoch=1,
        detect_anomaly=False,
        reload_dataloaders_every_n_epochs=1000,
        # resume_from_checkpoint="former3d/ckpts/last.ckpt",
    )
    trainer.fit(model)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="MTTU-Net-1228",
    #     entity='whz123',
    #     # track hyperparameters and run metadata
    #     config={
    #         "architecture": "MTTU-Net",
    #     })
    main_worker()
