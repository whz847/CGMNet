import os
import numpy as np
import nibabel as nib
import pyvista as pv
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from models.TransBraTS.TransBraTS_skipconnection import TransBraTS, IDH_ATRX_p19q_type_network
# from CBAM.model_resnet import ResidualNet3D
from CBAM.xiugai import ResidualNet3D
from models import criterions
from models.criterions import MultiTaskLossWrapper
from data.BraTS_IDH import BraTS
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix
import logging
from torch.utils.tensorboard import SummaryWriter
from models.WSI.model import CroMAM
# from models.WSI.Cromodels import CroMAM
from WSI_data.transform import get_transformation
from WSI_data.dataset import get_multi_dataloader_train, get_multi_dataloader_val
import torch.nn as nn
from models.criterions import idh_lmfloss, atrx_lmfloss, p19q_lmfloss


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        data_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        self.wsi_transform = get_transformation(mean=data_stats['mean'], std=data_stats['std'])
        self.model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        self.cbam3d_model = ResidualNet3D(18, 1000, 'CBAM3D')
        self.IDH_model = IDH_ATRX_p19q_type_network()
        self.CroMAM = CroMAM(
            backbone='resnet18',  # resnet50
            pretrained=True,
            device=torch.device("cuda:0"),
        )
        idh_criterion = getattr(criterions, 'idh_lmfloss')  # idh_focal_loss, idh_cross_entropy,idh_lmfloss
        atrx_criterion = getattr(criterions, 'atrx_lmfloss')
        p19q_criterion = getattr(criterions, 'p19q_lmfloss')
        # criterion = FocalLoss_seg()
        self.MTL = MultiTaskLossWrapper(3, loss_fn=[idh_criterion, atrx_criterion, p19q_criterion])
        self.config = config
        self.relu = nn.ReLU(inplace=True)
        self.alpha = 0.005
        self.tanh = nn.Tanh()
        self.modulation = "SGM"
        self.softmax = nn.Softmax(dim=1)
        self.noise_scale = 0.1

        # 在训练初始化阶段打开日志文件
        self.log_file_path = "gradient_log.csv"
        with open(self.log_file_path, 'w') as f:
            f.write("epoch,mri_model_gradient_std,mri_cbam_gradient_std,wsi_gradient_std\n")  # 写入表头

        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', config.experiment + config.date)
        log_file = log_dir + '.txt'
        self.log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(config):
            logging.info('{}={}'.format(arg, getattr(config, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(config.description))

        self.best_epoch = 0
        self.min_loss = 100.0
        self.best_acc = 0
        self.best_acc_1 = 0
        self.epoch = 1
        self.mriradio = 0.0
        self.wsiradio = 0.0

        self.checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                           config.experiment + config.date)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        writer = SummaryWriter()

        self.optimizer = None
        # 禁用自动优化
        self.automatic_optimization = False

        resume = ''
        if os.path.isfile(resume) and config.load:
            logging.info('loading checkpoint {}'.format(resume))
            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['en_state_dict'])
            self.cbam3d_model.load_state_dict(checkpoint['cbam_state_dict'])
            self.CroMAM.load_state_dict(checkpoint['wsi_state_dict'])
            self.IDH_model.load_state_dict(checkpoint['idh_state_dict'])
            logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                         .format(config.resume, config.start_epoch))
        else:
            logging.info('re-training!!!')
        # 新增对比学习投影头
        self.mri_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 128)  # 将MRI特征投影到128维
        )
        self.path_proj = nn.Linear(128, 128)  # 病理特征投影
        self.second_mri = nn.Linear(512, 128)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adagrad(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        return self.optimizer

    def on_train_epoch_start(self):
        self.epoch_train_logs = []
        """每个epoch开始时更新冻结状态"""
        current_epoch = self.current_epoch

    def _shared_step(self, batch):
        weight_IDH = torch.tensor([54, 66]).float()
        weight_ATRX = torch.tensor([30, 90]).float()
        weight_p19q = torch.tensor([20, 120]).float()
        x, idh, atrx, p19q, x_2, wsi_img = batch
        wsi_img = wsi_img.squeeze(0)

        # 对比学习特征处理
        mri_feat = self.mri_proj(encoder_output)  # [B, 128]
        path_feat = self.path_proj(wsi_feature)  # [B, 128]
        second_mri = self.second_mri(x_second)
        mri_feat = mri_feat + second_mri

        # 计算对比损失
        contrast_loss = self._contrastive_loss(mri_feat, path_feat)

        # 主任务预测
        mri_idh, wsi_idh, mri_atrx, wsi_atrx, mri_p19q, wsi_p19q, idh_out, atrx_out, p19q_out = self.IDH_model(x4_1,
                                                                                                               encoder_output,
                                                                                                               x_second,
                                                                                                               wsi_feature)

        # 主损失计算
        loss, idh_loss, atrx_loss, p19q_loss, idh_std, atrx_std, p19q_std, log_var_1, log_var_2, log_var_3 = self.MTL(
            [idh_out, atrx_out, p19q_out], [idh, atrx, p19q], [weight_IDH, weight_ATRX, weight_p19q])

        # 总损失组合
        total_loss = loss + 0.1 * contrast_loss  # + 0.1 * (loss_mri+loss_wsi)

        # 记录指标
        self.log('train/loss', total_loss)
        self.log('train/contrast_loss', contrast_loss)
        return total_loss

    def _contrastive_loss(self, feat1, feat2, temp=0.07):
        # 标准化特征
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)

        # 计算相似度矩阵
        logits = torch.mm(feat1, feat2.t()) / temp
        labels = torch.arange(feat1.size(0), device=feat1.device)
        return F.cross_entropy(logits, labels)

    def _freeze_components(self, freeze_mri=True, freeze_path=True, freeze_fusion=True):
        """冻结/解冻指定组件参数"""
        

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        # 强制前向传播所有组件（即使冻结）
        with torch.set_grad_enabled(True):
            loss = self._shared_step(batch)

        # 手动反向传播
        self.manual_backward(loss)

        mri_model_grad_std = 0.0
        mri_cbam_grad_std = 0.0
        wsi_grad_std = 0.0
        mri_count = 0
        wsi_count = 0
        # 处理MRI网络梯度
        for name, parms in self.model.named_parameters():
            if parms.grad is not None and len(parms.grad.size()) == 5:
                mri_model_grad_std += parms.grad.std().item()
                mri_count += 1
        if mri_count > 0:
            mri_model_grad_std /= mri_count

        for name, parms in self.cbam3d_model.named_parameters():
            if parms.grad is not None and len(parms.grad.size()) == 5:
                mri_cbam_grad_std += parms.grad.std().item()
        if mri_count > 0:
            mri_cbam_grad_std /= mri_count

        # 处理WSI网络梯度
        for name, parms in self.CroMAM.named_parameters():
            if parms.grad is not None and len(parms.grad.size()) == 4:
                wsi_grad_std += parms.grad.std().item()
                wsi_count += 1
        if wsi_count > 0:
            wsi_grad_std /= wsi_count

        # 将梯度写入日志文件
        with open(self.log_file_path, 'a') as f:
            f.write(f"{self.epoch},{mri_model_grad_std:.6f},{mri_cbam_grad_std:.6f},{wsi_grad_std:.6f}\n")

        opt.step()
        return loss

    def on_train_epoch_end(self):
        self.epoch += 1

    def on_validation_epoch_start(self):
        self.idh_probs = []
        self.idh_class = []
        self.idh_target = []
        self.atrx_probs = []
        self.atrx_class = []
        self.atrx_target = []
        self.p19q_probs = []
        self.p19q_class = []
        self.p19q_target = []
        self.epoch_valid_loss = 0.0
        self.epoch_idh_loss = 0.0
        self.epoch_atrx_loss = 0.0
        self.epoch_p19q_loss = 0.0

    def validation_step(self, data, batch_idx):

        x, idh, atrx, p19q, x_2, wsi_img = data
        wsi_img = wsi_img.squeeze(0)
        layer4 = self.cbam3d_model(x_2, 1)
        x1_1, x2_1, x3_1, x4_1, encoder_output, y, y1 = self.model(x, layer4)
        y = layer4 + y
        x_second = self.cbam3d_model(y, 2)
        wsi_feature = self.CroMAM(wsi_img)

        mri_idh, wsi_idh, mri_atrx, wsi_atrx, mri_p19q, wsi_p19q, idh_out, atrx_out, p19q_out = self.IDH_model(x4_1,
                                                                                                               encoder_output,
                                                                                                               x_second,
                                                                                                               wsi_feature)

        valid_loss, idh_loss, atrx_loss, p19q_loss, std_1, std_2, std_3, var_1, var_2, var_3 = self.MTL(
            [idh_out, atrx_out, p19q_out], [idh, atrx, p19q], [None, None, None])

        idh_pred = F.softmax(idh_out, 1)
        # idh_pred = idh_out.sigmoid()
        idh_pred_class = torch.argmax(idh_pred, dim=1)
        # idh_pred_class = (idh_pred > 0.5).float()
        self.idh_probs.append(idh_pred[0][1].cpu())
        # idh_probs.append(idh_pred[0])
        self.idh_class.append(idh_pred_class.item())
        self.idh_target.append(idh.item())

        atrx_pred = F.softmax(atrx_out, 1)
        atrx_pred_class = torch.argmax(atrx_pred, dim=1)
        self.atrx_probs.append(atrx_pred[0][1].cpu())
        self.atrx_class.append(atrx_pred_class.item())
        self.atrx_target.append(atrx.item())

        p19q_pred = F.softmax(p19q_out, 1)
        p19q_pred_class = torch.argmax(p19q_pred, dim=1)
        self.p19q_probs.append(p19q_pred[0][1].cpu())
        self.p19q_class.append(p19q_pred_class.item())
        self.p19q_target.append(p19q.item())

    def on_validation_epoch_end(self):
        # print("test")
        accuracy_idhv = accuracy_score(self.idh_target, self.idh_class)
        auc_idhv = roc_auc_score(self.idh_target, self.idh_probs)
        accuracy_atrxv = accuracy_score(self.atrx_target, self.atrx_class)
        auc_atrxv = roc_auc_score(self.atrx_target, self.atrx_probs)
        accuracy_p19qv = accuracy_score(self.p19q_target, self.p19q_class)
        auc_p19qv = roc_auc_score(self.p19q_target, self.p19q_probs)
        # 计算混淆矩阵
        idh_tn, idh_fp, idh_fn, idh_tp = confusion_matrix(self.idh_target, self.idh_class).ravel()
        atrx_tn, atrx_fp, atrx_fn, atrx_tp = confusion_matrix(self.atrx_target, self.atrx_class).ravel()
        p19q_tn, p19q_fp, p19q_fn, p19q_tp = confusion_matrix(self.p19q_target, self.p19q_class).ravel()
        # 计算特异性和敏感度
        specificity_idhv = idh_tn / (idh_tn + idh_fp)
        sensitivity_idhv = idh_tp / (idh_tp + idh_fn)
        specificity_atrxv = atrx_tn / (atrx_tn + atrx_fp)
        sensitivity_atrxv = atrx_tp / (atrx_tp + atrx_fn)
        specificity_p19qv = p19q_tn / (p19q_tn + p19q_fp)
        sensitivity_p19qv = p19q_tp / (p19q_tp + p19q_fn)
        # print("accuracy_idhv:",accuracy_idhv)
        if accuracy_atrxv + accuracy_idhv + accuracy_p19qv > self.best_acc:
            # min_loss = epoch_valid_loss
            self.best_acc = accuracy_atrxv + accuracy_idhv + accuracy_p19qv
            self.best_epoch = self.epoch
            logging.info('there is an improvement that update the metrics and save the best model.')
            logging.info(f'Epoch {self.epoch} | '
                         f'IDH_ACC: {accuracy_idhv:.5f}, IDH_AUC: {auc_idhv:.5f}, IDH_Sensitivity: {sensitivity_idhv:.5f}, IDH_Specificity: {specificity_idhv:.5f} | '
                         f'ATRX_ACC: {accuracy_atrxv:.5f}, ATRX_AUC: {auc_atrxv:.5f}, ATRX_Sensitivity: {sensitivity_atrxv:.5f}, ATRX_Specificity: {specificity_atrxv:.5f} | '
                         f'1p19q_ACC: {accuracy_p19qv:.5f}, 1p19q_AUC: {auc_p19qv:.5f}, 1p19q_Sensitivity: {sensitivity_p19qv:.5f}, 1p19q_Specificity: {specificity_p19qv:.5f}')

            file_name = os.path.join(self.checkpoint_dir,
                                     'model_' + str(self.epoch) + '_' + str(self.best_acc) + '_' + str(
                                         accuracy_idhv * 100) + '_' + str(accuracy_atrxv * 100) + '_' + str(
                                         accuracy_p19qv * 100) + '_' + str(
                                         auc_idhv) + '_' + str(auc_atrxv) + '_' + str(auc_p19qv) + '.pth')
            torch.save({
                'epoch': self.epoch,
                'en_state_dict': self.model.state_dict(),
                'cbam_state_dict': self.cbam3d_model.state_dict(),
                'wsi_state_dict': self.CroMAM.state_dict(),
                # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
                'idh_state_dict': self.IDH_model.state_dict(),
                'optim_dict': self.optimizer.state_dict(),
            },
                file_name)
        elif accuracy_atrxv + accuracy_idhv + accuracy_p19qv >= 2.60:
            acc_1 = accuracy_atrxv + accuracy_idhv + accuracy_p19qv
            file_name = os.path.join(self.checkpoint_dir, 'model_' + str(self.epoch) + '_' + str(acc_1) + '_' + str(
                accuracy_idhv * 100) + '_' + str(accuracy_atrxv * 100) + '_' + str(accuracy_p19qv * 100) + '_' + str(
                auc_idhv) + '_' + str(auc_atrxv) + '_' + str(auc_p19qv) + '.pth')
            torch.save({
                'epoch': self.epoch,
                'en_state_dict': self.model.state_dict(),
                'cbam_state_dict': self.cbam3d_model.state_dict(),
                'wsi_state_dict': self.CroMAM.state_dict(),
                # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
                'idh_state_dict': self.IDH_model.state_dict(),
                'optim_dict': self.optimizer.state_dict(),
            },
                file_name)
        # elif auc_p19qv > 0.8:
        #     acc_2 = accuracy_atrxv + accuracy_idhv + accuracy_p19qv
        #     file_name = os.path.join(self.checkpoint_dir,
        #                              'auc_p19qv大于0.8_model_' + str(self.epoch) + '_' + str(acc_2) + '_' + str(
        #                                  accuracy_idhv * 100) + '_' + str(accuracy_atrxv * 100) + '_' + str(
        #                                  accuracy_p19qv * 100) + '_' + str(auc_idhv) + '_' + str(auc_atrxv) + '_' + str(
        #                                  auc_p19qv) + '.pth')
        #     torch.save({
        #         'epoch': self.epoch,
        #         'en_state_dict': self.model.state_dict(),
        #         'cbam_state_dict': self.cbam3d_model.state_dict(),
        #         # 'cbam2_state_dict': DDP_model['cbam2'].state_dict(),
        #         'idh_state_dict': self.IDH_model.state_dict(),
        #         'optim_dict': self.optimizer.state_dict(),
        #     },
        #         file_name)
        logging.info(
            'Epoch:{}[best_epoch:{} ||best_acc:{:.5f}| epoch_valid_loss:{:.5f} |idh_loss: {:.5f} | atrx_loss: {:.5f} | p19q_loss: {:.5f} || idhv_acc: {:.5f} | idhv_auc:{:.5f} | idhv_sens:{:.5f} | idhv_spec:{:.5f} | atrxv_acc: {:.5f} | atrxv_auc:{:.5f} | atrxv_sens:{:.5f} | atrxv_spec:{:.5f} | p19qv_acc: {:.5f} | p19qv_auc:{:.5f} | p19qv_sens:{:.5f} | p19qv_spec:{:.5f}'
            .format(self.epoch, self.best_epoch, self.best_acc, self.epoch_valid_loss, self.epoch_idh_loss,
                    self.epoch_atrx_loss,
                    self.epoch_p19q_loss, accuracy_idhv, auc_idhv, sensitivity_idhv, specificity_idhv,
                    accuracy_atrxv, auc_atrxv, sensitivity_atrxv, specificity_atrxv, accuracy_p19qv, auc_p19qv,
                    sensitivity_p19qv, specificity_p19qv))

    def epoch_end(self, logs, prefix):
        keys = set([key for log in logs for key in log])
        results = {key: [] for key in keys}
        for log in logs:
            for key, value in log.items():
                results[key].append(value)
        logs = {f"{prefix}/{key}": np.nanmean(results[key]) for key in keys}
        self.log_dict(logs, rank_zero_only=True)
        # if prefix == 'val':
        #     self.log('val_voxel_loss_fine', logs["val/voxel_loss_fine"], rank_zero_only=True)

    def train_dataloader(self):
        return self.dataloader("train", augment=True)

    def val_dataloader(self):
        val_loader = get_multi_dataloader_val(self.config, self.wsi_transform)
        valid_list = os.path.join(self.config.root, self.config.valid_dir, self.config.valid_file)
        valid_root = os.path.join(self.config.root, self.config.valid_dir)
        valid_set = BraTS(valid_list, self.config, valid_root, 'valid', wsi_loader=val_loader)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=0,
                                                   pin_memory=True)
        return valid_loader

    def dataloader(self, split, augment=False):
        train_loader = get_multi_dataloader_train(self.config, self.wsi_transform)
        train_list = os.path.join(self.config.root, self.config.train_dir, self.config.train_file)
        train_root = os.path.join(self.config.root, self.config.train_dir)
        train_set = BraTS(train_list, self.config, train_root, self.config.mode, wsi_loader=train_loader)

        return torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=1,
            drop_last=True,
            num_workers=self.config.num_workers
        )

    def log_args(self, log_file):
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
