# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py

import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.TransBraTS.TransBraTS_skipconnection import TransBraTS,Decoder_modual,IDH_ATRX_p19q_type_network,Grade_netwoek
# from CBAM.model_resnet import ResidualNet3D
from CBAM.xiugai import ResidualNet3D
from models.unet import UNet3D
import torch.distributed as dist
from models import criterions
from contextlib import nullcontext
from data.BraTS_IDH import BraTS
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from utils.pcgrad import PCGrad
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from models.criterions import MultiTaskLossWrapper
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import confusion_matrix
import wandb
import random

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
parser.add_argument('--root', default='/mnt/K/WHZ/datasets/BraTS2020 T+V/', type=str)

parser.add_argument('--train_dir', default='BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', type=str)

parser.add_argument('--valid_dir', default='BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData', type=str)

parser.add_argument('--test_dir', default='BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train_label_add.txt', type=str)#IDH_all.txt

parser.add_argument('--valid_file', default='valid_label.txt', type=str) #IDH_test.txt

parser.add_argument('--test_file', default='test_label.txt', type=str)#IDH_test.txt

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

parser.add_argument('--criterion', default='softmax_dice', type=str)#softmax_dice常见于医学影像分割任务中，它是一种结合了Softmax交叉熵损失和Dice损失的组合损失函数。

parser.add_argument('--num_class', default=3, type=int)#4

parser.add_argument('--seed', default=1234, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1', type=str)

parser.add_argument('--num_workers', default=4, type=int)#8

parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=1000, type=int)

parser.add_argument('--save_freq', default=750, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()

def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date+"test")
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    # 通过这些设置，可以保证在分布式训练中每个进程使用不同的随机数种子，从而使实验具备可重现性，并确保每个进程在GPU上都使用正确的设备进行计算。
    torch.distributed.init_process_group('nccl')#,world_size=2, rank=args.local_rank
    torch.cuda.set_device(args.local_rank)
    rank = torch.distributed.get_rank()
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    cbam3d_model=ResidualNet3D(18,1000,'CBAM3D')#这儿的num_classes随便写，因为后面resnet全连接层用的部分注释了，用不到
    # cbam3d_model2 = ResidualNet3D(18, 1000, 'CBAM3D')
    IDH_model = IDH_ATRX_p19q_type_network()

    # criterion = getattr(criterions, args.criterion)  # args.criterion
    idh_criterion = getattr(criterions, 'idh_lmfloss')  # idh_focal_loss, idh_cross_entropy,idh_lmfloss
    atrx_criterion = getattr(criterions, 'atrx_lmfloss')
    p19q_criterion = getattr(criterions, 'p19q_lmfloss')
    # criterion = FocalLoss_seg()
    MTL = MultiTaskLossWrapper(3, loss_fn=[idh_criterion,atrx_criterion,p19q_criterion])

    """下面这段代码涉及到了分布式训练中的模型并行和数据并行的概念。展示了如何将多个模型进行分布式并行训练，并使用DistributedDataParallel封装模型以便在分布式环境中进行训练。"""
    nets = {
        'en': torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(args.local_rank),
        'cbam':torch.nn.SyncBatchNorm.convert_sync_batchnorm(cbam3d_model).cuda(args.local_rank),
        # 'cbam2': torch.nn.SyncBatchNorm.convert_sync_batchnorm(cbam3d_model2).cuda(args.local_rank),
        'idh': torch.nn.SyncBatchNorm.convert_sync_batchnorm(IDH_model).cuda(args.local_rank),
        'mtl': MTL.cuda(args.local_rank)
    }
    param = [p for v in nets.values() for p in list(v.parameters())]

    DDP_model = {
        'en': nn.parallel.DistributedDataParallel(nets['en'], device_ids=[args.local_rank],
                                                  output_device=args.local_rank,
                                                  find_unused_parameters=True),
        'cbam':nn.parallel.DistributedDataParallel(nets['cbam'],device_ids=[args.local_rank],
                                                   output_device=args.local_rank,
                                                   find_unused_parameters=True),
        # 'cbam2': nn.parallel.DistributedDataParallel(nets['cbam2'], device_ids=[args.local_rank],
        #                                             output_device=args.local_rank,
        #                                             find_unused_parameters=True),
        'idh': nn.parallel.DistributedDataParallel(nets['idh'], device_ids=[args.local_rank],
                                                   output_device=args.local_rank,
                                                   find_unused_parameters=True),
        'mtl': nn.parallel.DistributedDataParallel(nets['mtl'], device_ids=[args.local_rank],
                                                   output_device=args.local_rank,
                                                   find_unused_parameters=True)
    }


    optimizer = torch.optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                      args.experiment + args.date+"test")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        writer = SummaryWriter()

    resume = ''

    # writer = SummaryWriter()

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        DDP_model['en'].load_state_dict(checkpoint['en_state_dict'])
        DDP_model['cbam'].load_state_dict(checkpoint['cbam_state_dict'])
        # DDP_model['cbam2'].load_state_dict(checkpoint['cbam2_state_dict'])
        DDP_model['idh'].load_state_dict(checkpoint['idh_state_dict'])
        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = BraTS(train_list, train_root, args.mode)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    logging.info('Samples for train = {}'.format(len(train_set)))

    num_gpu = (len(args.gpu) + 1) // 2

    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root, 'valid')
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logging.info('Samples for valid = {}'.format(len(valid_set)))

    start_time = time.time()
    torch.set_grad_enabled(True)

    best_epoch = 0
    min_loss = 100.0
    best_acc=0
    best_acc_1=0

    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
        DDP_model['en'].train()
        DDP_model['cbam'].train()
        # DDP_model['cbam2'].train()
        DDP_model['idh'].train()
        DDP_model['mtl'].train()
        train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        start_epoch = time.time()

        epoch_train_loss = 0.0
        epoch_train_idh_loss = 0.0
        epoch_train_atrx_loss=0.0
        epoch_train_p19q_loss = 0.0

        for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
            optimizer.zero_grad()
            x, idh, atrx, p19q, x_2 = data
            # x, idh, atrx, p19q,x_2,x_2_2 = data
            x = x.cuda(args.local_rank, non_blocking=True)
            x_2 = x_2.cuda(args.local_rank, non_blocking=True)
            # x_2_2 = x_2_2.cuda(args.local_rank, non_blocking=True)
            # target = target.cuda(args.local_rank, non_blocking=True)
            idh = idh.cuda(args.local_rank, non_blocking=True)
            atrx = atrx.cuda(args.local_rank, non_blocking=True)
            p19q = p19q.cuda(args.local_rank, non_blocking=True)
            weight_IDH = torch.tensor([101, 67]).float().cuda(args.local_rank, non_blocking=True)#57, 109
            weight_ATRX=torch.tensor([50,118]).float().cuda(args.local_rank, non_blocking=True)#33,133
            weight_p19q=torch.tensor([40,150]).float().cuda(args.local_rank, non_blocking=True)#13,153

            layer4=DDP_model['cbam'](x_2,1)
            x1_1, x2_1, x3_1, x4_1, encoder_output,y = DDP_model['en'](x,layer4)
            y=layer4+y
            x_second = DDP_model['cbam'](y,2)
            # x_second2 = DDP_model['cbam2'](x_2_2)
            # idh_out, atrx_out, p19q_out=DDP_model['idh'](x4_1, encoder_output,x_second,x_second2)
            idh_out, atrx_out, p19q_out = DDP_model['idh'](x4_1, encoder_output, x_second)

            loss,idh_loss,atrx_loss,p19q_loss,idh_std,atrx_std,p19q_std,log_var_1, log_var_2,log_var_3= DDP_model['mtl']([idh_out,atrx_out,p19q_out], [idh,atrx,p19q], [weight_IDH,weight_ATRX,weight_p19q])
            # 下面这段代码是用于在分布式训练过程中对损失和统计量进行全局归约和汇总，以便更好地跟踪和记录模型在整个训练数据集上的性能。
            reduce_idh_loss = all_reduce_tensor(idh_loss, world_size=num_gpu).data.cpu().numpy()
            reduce_atrx_loss = all_reduce_tensor(atrx_loss, world_size=num_gpu).data.cpu().numpy()
            reduce_p19q_loss = all_reduce_tensor(p19q_loss, world_size=num_gpu).data.cpu().numpy()
            idh_std = all_reduce_tensor(idh_std, world_size=num_gpu).data.cpu().numpy()
            atrx_std = all_reduce_tensor(atrx_std, world_size=num_gpu).data.cpu().numpy()
            p19q_std = all_reduce_tensor(p19q_std, world_size=num_gpu).data.cpu().numpy()
            idh_vars = all_reduce_tensor(log_var_1, world_size=num_gpu).data.cpu().numpy()
            atrx_vars = all_reduce_tensor(log_var_2, world_size=num_gpu).data.cpu().numpy()
            p19q_vars = all_reduce_tensor(log_var_3, world_size=num_gpu).data.cpu().numpy()

            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
            epoch_train_loss += reduce_loss / len(train_loader)
            epoch_train_idh_loss += reduce_idh_loss / len(train_loader)
            epoch_train_atrx_loss += reduce_atrx_loss / len(train_loader)
            epoch_train_p19q_loss += reduce_p19q_loss / len(train_loader)

            if args.local_rank == 0:
                logging.info('Epoch: {}_Iter:{}  loss: {:.5f}  idh_loss: {:.5f} atrx_loss: {:.5f} p19q_loss: {:.5f} ||idh_std:{:.4f} atrx_std:{:.4f} p19q_std:{:.4f} idh_vars:{:.4f} atrx_vars:{:.4f} p19q_vars:{:.4f}'
                                    .format(epoch,i, reduce_loss,reduce_idh_loss,reduce_atrx_loss,reduce_p19q_loss,idh_std,atrx_std,p19q_std,idh_vars,atrx_vars,p19q_vars))
            loss.backward()
            optimizer.step()

        idh_probs = []
        idh_class = []
        idh_target = []
        atrx_probs = []
        atrx_class = []
        atrx_target = []
        p19q_probs = []
        p19q_class = []
        p19q_target = []
        with torch.no_grad():
            DDP_model['en'].eval()
            DDP_model['cbam'].eval()
            # DDP_model['cbam2'].eval()
            DDP_model['idh'].eval()
            DDP_model['mtl'].eval()
            epoch_valid_loss = 0.0
            epoch_idh_loss = 0.0
            epoch_atrx_loss = 0.0
            epoch_p19q_loss = 0.0
            for i, data in enumerate(valid_loader):
                # [t.cuda(args.local_rank, non_blocking=True) for t in data]
                # x,idh,atrx,p19q,x_2,x_2_2 = data
                x, idh, atrx, p19q, x_2= data
                x = x.cuda(args.local_rank, non_blocking=True)
                x_2 = x_2.cuda(args.local_rank, non_blocking=True)
                # x_2_2 = x_2_2.cuda(args.local_rank, non_blocking=True)
                # target = target.cuda(args.local_rank, non_blocking=True)
                idh = idh.cuda(args.local_rank, non_blocking=True)
                atrx = atrx.cuda(args.local_rank, non_blocking=True)
                p19q = p19q.cuda(args.local_rank, non_blocking=True)

                layer4=DDP_model['cbam'](x_2,1)
                x1_1, x2_1, x3_1, x4_1, encoder_output, y = DDP_model['en'](x,layer4)
                y=layer4+y
                x_second = DDP_model['cbam'](y,2)
                # x_second2 = DDP_model['cbam2'](x_2_2)
                # idh_out,atrx_out,p19q_out = DDP_model['idh'](encoder_outs[3], encoder_outs[4],x_second,x_second2)
                idh_out, atrx_out, p19q_out = DDP_model['idh'](x4_1,encoder_output,x_second)

                weight_IDH = torch.tensor([101, 67]).float().cuda(args.local_rank, non_blocking=True)  # 57, 109
                weight_ATRX = torch.tensor([50, 118]).float().cuda(args.local_rank, non_blocking=True)  # 33,133
                weight_p19q = torch.tensor([40, 150]).float().cuda(args.local_rank, non_blocking=True)  # 13,153

                valid_loss, idh_loss, atrx_loss, p19q_loss,std_1,std_2, std_3,var_1, var_2, var_3=DDP_model['mtl']([idh_out, atrx_out, p19q_out], [idh, atrx, p19q],[weight_IDH,weight_ATRX,weight_p19q])
                epoch_valid_loss += valid_loss / len(valid_loader)

                epoch_idh_loss += idh_loss / len(valid_loader)
                epoch_atrx_loss += atrx_loss / len(valid_loader)
                epoch_p19q_loss += p19q_loss / len(valid_loader)

                idh_pred = F.softmax(idh_out, 1)
                # idh_pred = idh_out.sigmoid()
                idh_pred_class = torch.argmax(idh_pred, dim=1)
                # idh_pred_class = (idh_pred > 0.5).float()
                idh_probs.append(idh_pred[0][1].cpu())
                # idh_probs.append(idh_pred[0])
                idh_class.append(idh_pred_class.item())
                idh_target.append(idh.item())

                atrx_pred = F.softmax(atrx_out, 1)
                atrx_pred_class = torch.argmax(atrx_pred, dim=1)
                atrx_probs.append(atrx_pred[0][1].cpu())
                atrx_class.append(atrx_pred_class.item())
                atrx_target.append(atrx.item())

                p19q_pred = F.softmax(p19q_out, 1)
                p19q_pred_class = torch.argmax(p19q_pred, dim=1)
                p19q_probs.append(p19q_pred[0][1].cpu())
                p19q_class.append(p19q_pred_class.item())
                p19q_target.append(p19q.item())

            accuracy_idhv = accuracy_score(idh_target, idh_class)
            auc_idhv = roc_auc_score(idh_target, idh_probs)
            accuracy_atrxv = accuracy_score(atrx_target, atrx_class)
            auc_atrxv = roc_auc_score(atrx_target, atrx_probs)
            accuracy_p19qv = accuracy_score(p19q_target, p19q_class)
            auc_p19qv = roc_auc_score(p19q_target, p19q_probs)
            # 计算混淆矩阵
            idh_tn, idh_fp, idh_fn, idh_tp = confusion_matrix(idh_target, idh_class).ravel()
            atrx_tn, atrx_fp, atrx_fn, atrx_tp = confusion_matrix(atrx_target, atrx_class).ravel()
            p19q_tn, p19q_fp, p19q_fn, p19q_tp = confusion_matrix(p19q_target, p19q_class).ravel()
            # 计算特异性和敏感度
            specificity_idhv = idh_tn / (idh_tn + idh_fp)
            sensitivity_idhv = idh_tp / (idh_tp + idh_fn)
            specificity_atrxv = atrx_tn / (atrx_tn + atrx_fp)
            sensitivity_atrxv = atrx_tp / (atrx_tp + atrx_fn)
            specificity_p19qv = p19q_tn / (p19q_tn + p19q_fp)
            sensitivity_p19qv = p19q_tp / (p19q_tp + p19q_fn)

            if args.local_rank == 0:

                if accuracy_atrxv+accuracy_idhv+accuracy_p19qv>=best_acc: #min_loss >= epoch_valid_loss:
                    # min_loss = epoch_valid_loss
                    best_acc=accuracy_atrxv+accuracy_idhv+accuracy_p19qv
                    best_epoch = epoch
                    logging.info('there is an improvement that update the metrics and save the best model.')
                    logging.info(f'Epoch {epoch} | '
                                 f'IDH_ACC: {accuracy_idhv:.5f}, IDH_AUC: {auc_idhv:.5f}, IDH_Sensitivity: {sensitivity_idhv:.5f}, IDH_Specificity: {specificity_idhv:.5f} | '
                                 f'ATRX_ACC: {accuracy_atrxv:.5f}, ATRX_AUC: {auc_atrxv:.5f}, ATRX_Sensitivity: {sensitivity_atrxv:.5f}, ATRX_Specificity: {specificity_atrxv:.5f} | '
                                 f'1p19q_ACC: {accuracy_p19qv:.5f}, 1p19q_AUC: {auc_p19qv:.5f}, 1p19q_Sensitivity: {sensitivity_p19qv:.5f}, 1p19q_Specificity: {specificity_p19qv:.5f}')

                    file_name = os.path.join(checkpoint_dir, 'model_'+str(epoch)+'_'+str(best_acc)+'_'+str(accuracy_idhv*100)+'_'+str(accuracy_atrxv*100)+'_'+str(accuracy_p19qv*100)+'_'+str(auc_idhv)+'_'+str(auc_atrxv)+'_'+str(auc_p19qv)+'.pth')
                    torch.save({
                        'epoch': epoch,
                        'en_state_dict': DDP_model['en'].state_dict(),
                        'cbam_state_dict': DDP_model['cbam'].state_dict(),
                        # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
                        'idh_state_dict': DDP_model['idh'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)
                # elif epoch>200 and epoch<600 and epoch%2==0:
                #     acc_1=accuracy_atrxv+accuracy_idhv+accuracy_p19qv
                #     file_name=os.path.join(checkpoint_dir,'model_'+str(epoch)+'_'+str(acc_1)+'_'+str(accuracy_idhv*100)+'_'+str(accuracy_atrxv*100)+'_'+str(accuracy_p19qv*100)+'_'+str(auc_idhv)+'_'+str(auc_atrxv)+'_'+str(auc_p19qv)+'.pth')
                #     torch.save({
                #         'epoch': epoch,
                #         'en_state_dict': DDP_model['en'].state_dict(),
                #         'cbam_state_dict': DDP_model['cbam'].state_dict(),
                #         # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
                #         'idh_state_dict': DDP_model['idh'].state_dict(),
                #         'optim_dict': optimizer.state_dict(),
                #     },
                #         file_name)
                # elif auc_p19qv>0.8:
                #     acc_2 = accuracy_atrxv + accuracy_idhv + accuracy_p19qv
                #     file_name=os.path.join(checkpoint_dir,'auc_p19qv大于0.8_model_'+str(epoch)+'_'+str(acc_2)+'_'+str(accuracy_idhv*100)+'_'+str(accuracy_atrxv*100)+'_'+str(accuracy_p19qv*100)+'_'+str(auc_idhv)+'_'+str(auc_atrxv)+'_'+str(auc_p19qv)+'.pth')
                #     torch.save({
                #         'epoch': epoch,
                #         'en_state_dict': DDP_model['en'].state_dict(),
                #         'cbam_state_dict': DDP_model['cbam'].state_dict(),
                #         # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
                #         'idh_state_dict': DDP_model['idh'].state_dict(),
                #         'optim_dict': optimizer.state_dict(),
                #     },
                #         file_name)
                logging.info(
                    'Epoch:{}[best_epoch:{} ||best_acc:{:.5f}| epoch_valid_loss:{:.5f} |idh_loss: {:.5f} | atrx_loss: {:.5f} | p19q_loss: {:.5f} || idhv_acc: {:.5f} | idhv_auc:{:.5f} | idhv_sens:{:.5f} | idhv_spec:{:.5f} | atrxv_acc: {:.5f} | atrxv_auc:{:.5f} | atrxv_sens:{:.5f} | atrxv_spec:{:.5f} | p19qv_acc: {:.5f} | p19qv_auc:{:.5f} | p19qv_sens:{:.5f} | p19qv_spec:{:.5f}'
                    .format(epoch, best_epoch, best_acc, epoch_valid_loss, epoch_idh_loss, epoch_atrx_loss,
                            epoch_p19q_loss, accuracy_idhv, auc_idhv, sensitivity_idhv, specificity_idhv,
                            accuracy_atrxv, auc_atrxv, sensitivity_atrxv, specificity_atrxv, accuracy_p19qv, auc_p19qv,
                            sensitivity_p19qv, specificity_p19qv))

        end_epoch = time.time()
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 3) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'en_state_dict': DDP_model['en'].state_dict(),
                    'cbam_state_dict': DDP_model['cbam'].state_dict(),
                    # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
                    'idh_state_dict': DDP_model['idh'].state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

            writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('epoch_train_loss:', epoch_train_loss, epoch)
            writer.add_scalar('epoch_valid_loss:', epoch_valid_loss, epoch)
            writer.add_scalar('idh_loss:', epoch_train_idh_loss, epoch)
            writer.add_scalar('atrx_loss:', epoch_train_atrx_loss, epoch)
            writer.add_scalar('p19q_loss:', epoch_train_p19q_loss, epoch)
            writer.add_scalar('valid_idh_loss:', epoch_idh_loss, epoch)
            writer.add_scalar('valid_atrx_loss:', epoch_atrx_loss, epoch)
            writer.add_scalar('valid_p19q_loss:', epoch_p19q_loss, epoch)

        # wandb.log({"epoch_train_loss":epoch_train_loss,"train_idh_loss":epoch_train_idh_loss,"train_atrx_loss":epoch_train_atrx_loss,"train_p19q_loss":epoch_train_p19q_loss})
        # wandb.log({"epoch_valid_loss":epoch_valid_loss,"valid_idh_loss":epoch_idh_loss,"valid_atrx_loss":epoch_atrx_loss,"valid_p19q_loss":epoch_p19q_loss})
        # wandb.log({"best_acc":accuracy_idhv+accuracy_atrxv+accuracy_p19qv,"idhv_acc":accuracy_idhv,"idhv_auc":auc_idhv,"atrxv_acc":accuracy_atrxv,"atrxv_auc":auc_atrxv,"p19qv_acc":accuracy_p19qv,"p19qv_auc":auc_p19qv})
        # wandb.log({"idhv_sens":sensitivity_idhv,"idhv_spec":specificity_idhv,"atrxv_sens":sensitivity_atrxv,"atrxv_spec":specificity_atrxv,"p19q_sens":sensitivity_p19qv,"p19q_spec":specificity_p19qv})

        # 下面这段代码用于估计当前训练轮次的时间消耗以及预估剩余训练时间，并输出到日志中。
        if args.local_rank == 0:
            epoch_time_minute = (end_epoch - start_epoch) / 60
            remaining_time_hour = (args.end_epoch - epoch - 1) * epoch_time_minute / 60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:
        writer.close()

        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'en_state_dict': DDP_model['en'].state_dict(),
            'cbam_state_dict': DDP_model['cbam'].state_dict(),
            # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
            'idh_state_dict': DDP_model['idh'].state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


"""下面adjust_learning_rate方法：
学习率的更新公式为init_lr * np.power(1-(epoch) / max_epoch, power)，其中np.power()函数用于计算幂次方。这个公式会根据当前的训练轮数和总的训练轮数来动态地计算学习率的衰减程度。
最后，学习率被舍入到8位小数，并赋值给param_group['lr']，从而实现对学习率的更新。
通过这种方式，在训练过程中可以根据训练轮数的进展来自动调整学习率。这对于优化模型的训练效果和加快收敛速度都是非常有帮助的。"""


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):  # power：学习率衰减的指数
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


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