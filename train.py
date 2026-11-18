import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
import time
import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast

import kornia
import utils.visualization as visual
from utils import dataloader
from utils.utils import clip_gradient
from utils.utils import cross_entropy_loss_RCF
from torch.optim import lr_scheduler
from utils.metrics import Evaluator
from network import UNet
from network.Deeplabv3_plus import deeplabv3
from network import HRNet
from network.PSPNet import Pspnet
from network.ABCNet import ABCNet
from network.CMTFNet import CMTFNet
from network import MCCA
from network import CGNet
from network import ENet
from network import DenseASPP
from network import SegNet
from network import BuildFormer
from network.UANet.UANet_final import UANet_final_Res50
from network import DSNet
from network.UNetFormer import UNetFormer
from network.DBBANet.model import DBBANet, DBBANet_CTRCL, DBBANet_Enhanced
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# 选择 AMP 精度
# AMP_DTYPE = torch.bfloat16 if (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16
AMP_DTYPE = torch.float16  # 强制用 FP16，避免 BF16 的 bilinear interpolate 未实现


def log_and_print(message, log_file=None):
    """同时打印到控制台和写入日志文件"""
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")


def _to_numpy_mask(logits_or_tuple, y_tensor):
    """将模型输出转换为 numpy 的整型分割掩码（先 argmax 再搬 CPU，避免搬整块 logits）"""
    if isinstance(logits_or_tuple, (list, tuple)):
        logits = logits_or_tuple[0]
    else:
        logits = logits_or_tuple
    pred = logits.argmax(1)  # [N,H,W]
    target = y_tensor
    return target.detach().cpu().numpy(), pred.detach().cpu().numpy()

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def train_UNetFormer(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, iters, opt, epoch, scaler, log_file=None):
    global best_iou
    global best_epoch

    epoch_loss = 0.0
    length = 0
    net.train(True)

    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda(non_blocking=True)
        Y = mask.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=AMP_DTYPE):
            output = net(X)  # 期望返回 [out_main, out_aux]
            # 原版是两个输出都监督；若显存吃紧，可只监督主分支
            loss = criterion(output[0], Y.long()) + 0.3 * criterion(output[1], Y.long())

        scaler.scale(loss).backward()
        clip_gradient(optimizer, opt.clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += float(loss.detach().item())
        length += 1

        with torch.no_grad():
            target_np, pred_np = _to_numpy_mask(output, Y)
            Eva_train.add_batch(target_np, pred_np)

        del output, loss, X, Y
        if i % 50 == 0:
            torch.cuda.empty_cache()

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    Precision = Eva_train.Precision()
    Recall = Eva_train.Recall()
    train_loss = epoch_loss / max(1, length)

    print('Epoch [%d/%d], Loss: %.4f, \n[Training] IOU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f'
          % (epoch, opt.epoch, train_loss, IOU[1], F1, Precision, Recall))
    print("Start validating!")

    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda(non_blocking=True)
            Y = mask.cuda(non_blocking=True)
            with autocast(dtype=AMP_DTYPE):
                output = net(X)
            target_np, pred_np = _to_numpy_mask(output, Y)
            Eva_val.add_batch(target_np, pred_np)
            del output, X, Y

    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()
    Precision = Eva_val.Precision()
    Recall = Eva_val.Recall()

    print('Epoch [%d/%d], \n[Validating] IOU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f'
          % (epoch, opt.epoch, IOU[1], F1, Precision, Recall))

    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best %s Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (opt.model_name, IOU[1], F1, best_epoch))
        model_path = os.path.join(opt.save_path, f'{opt.model_name}_{opt.data_name}_best.pth')
        torch.save(best_net, model_path)
        print(f'模型已保存到: {model_path}')
    print('%s Model Iou :%.4f; Best IoU is :%.4f, Best epoch is :%d' % (opt.model_name, new_iou, best_iou, best_epoch))

    return train_loss, new_iou


def train_UANet(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, iters, opt, epoch, scaler, log_file=None):
    global best_iou
    global best_epoch
    epoch_loss = 0.0
    length = 0
    net.train(True)

    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda(non_blocking=True)
        Y = mask.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=AMP_DTYPE):
            output = net(X)   # 期望返回 [o0, o1, o2, o3, o4]，o4 为主分支
            # 仅主分支监督，显存友好；如需原逻辑可改为原来的五项相加
            loss = criterion(output[4], Y.long())
            # loss = (criterion(output[0], Y.long()) + criterion(output[1], Y.long())
            #         + criterion(output[2], Y.long()) + criterion(output[3], Y.long())
            #         + criterion(output[4], Y.long()))

        scaler.scale(loss).backward()
        clip_gradient(optimizer, opt.clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += float(loss.detach().item())
        length += 1

        with torch.no_grad():
            pred = output[4].argmax(1)
            Eva_train.add_batch(Y.detach().cpu().numpy(), pred.detach().cpu().numpy())

        del output, loss, X, Y, pred
        if i % 50 == 0:
            torch.cuda.empty_cache()

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    train_loss = epoch_loss / max(1, length)

    print('Epoch [%d/%d], Loss: %.4f, \n[Training] IOU: %.4f, F1: %.4f'
          % (epoch, opt.epoch, train_loss, IOU[1], F1))
    print("Start validating!")

    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda(non_blocking=True)
            Y = mask.cuda(non_blocking=True)
            with autocast(dtype=AMP_DTYPE):
                output = net(X)
            pred = output[4].argmax(1)
            Eva_val.add_batch(Y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            del output, X, Y, pred

    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()
    print('Epoch [%d/%d],\n[Validating] IOU: %.4f, F1: %.4f'
          % (epoch, opt.epoch, IOU[1], F1))
    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best %s Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (opt.model_name, IOU[1], F1, best_epoch))
        model_path = os.path.join(opt.save_path, f'{opt.model_name}_{opt.data_name}_best.pth')
        torch.save(best_net, model_path)
        print(f'模型已保存到: {model_path}')
    print('%s Model Iou :%.4f; Best IoU is :%.4f, Best epoch is :%d' % (opt.model_name, new_iou, best_iou, best_epoch))

    return train_loss, new_iou


def train_DSNet(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, iters, opt, epoch, scaler, log_file=None):
    global best_iou
    global best_epoch
    epoch_loss = 0.0
    length = 0
    net.train(True)

    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda(non_blocking=True)
        Y = mask.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=AMP_DTYPE):
            output = net(X)
            # 只监督主输出（output[0]），避免对多个上采样分支反传导致显存飙升
            loss = criterion(output[0], Y.long())

        scaler.scale(loss).backward()
        clip_gradient(optimizer, opt.clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += float(loss.detach().item())
        length += 1

        with torch.no_grad():
            pred = output[0].argmax(1)
            Eva_train.add_batch(Y.detach().cpu().numpy(), pred.detach().cpu().numpy())

        del output, loss, X, Y, pred
        if i % 50 == 0:
            torch.cuda.empty_cache()

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    train_loss = epoch_loss / max(1, length)

    print('Epoch [%d/%d], Loss: %.4f, \n[Training] IOU: %.4f, F1: %.4f'
          % (epoch, opt.epoch, train_loss, IOU[1], F1))
    print("Start validating!")

    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda(non_blocking=True)
            Y = mask.cuda(non_blocking=True)
            with autocast(dtype=AMP_DTYPE):
                output = net(X)
            pred = output[0].argmax(1)
            Eva_val.add_batch(Y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            del output, X, Y, pred

    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()
    print('Epoch [%d/%d],\n[Validating] IOU: %.4f, F1: %.4f'
          % (epoch, opt.epoch, IOU[1], F1))
    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best %s Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (opt.model_name, IOU[1], F1, best_epoch))
        model_path = os.path.join(opt.save_path, f'{opt.model_name}_{opt.data_name}_best.pth')
        torch.save(best_net, model_path)
        print(f'模型已保存到: {model_path}')
    print('%s Model Iou :%.4f; Best IoU is :%.4f, Best epoch is :%d' % (opt.model_name, new_iou, best_iou, best_epoch))

    return train_loss, new_iou


def train_ABC(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, iters, opt, epoch, scaler, log_file=None):
    global best_iou
    global best_epoch
    epoch_loss = 0.0
    length = 0
    net.train(True)

    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda(non_blocking=True)
        Y = mask.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=AMP_DTYPE):
            output = net(X)  # [o0, o1, o2]
            # 只监督主分支（o0）
            loss = criterion(output[0], Y.long())
            # 如需多分支： + 0.3*criterion(output[1], ...) + 0.3*criterion(output[2], ...)

        scaler.scale(loss).backward()
        clip_gradient(optimizer, opt.clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += float(loss.detach().item())
        length += 1

        with torch.no_grad():
            pred = output[0].argmax(1)
            Eva_train.add_batch(Y.detach().cpu().numpy(), pred.detach().cpu().numpy())

        del output, loss, X, Y, pred
        if i % 50 == 0:
            torch.cuda.empty_cache()

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    train_loss = epoch_loss / max(1, length)

    print('Epoch [%d/%d], Loss: %.4f, \n[Training] IOU: %.4f, F1: %.4f'
          % (epoch, opt.epoch, train_loss, IOU[1], F1))
    print("Start validating!")

    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda(non_blocking=True)
            Y = mask.cuda(non_blocking=True)
            with autocast(dtype=AMP_DTYPE):
                output = net(X)
            pred = output[0].argmax(1)
            Eva_val.add_batch(Y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            del output, X, Y, pred

    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()
    print('Epoch [%d/%d], \n[Validating] IOU: %.4f, F1: %.4f'
          % (epoch, opt.epoch, IOU[1], F1))
    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best %s Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (opt.model_name, IOU[1], F1, best_epoch))
        model_path = os.path.join(opt.save_path, f'{opt.model_name}_{opt.data_name}_best.pth')
        torch.save(best_net, model_path)
        print(f'模型已保存到: {model_path}')
    print('%s Model Iou :%.4f; Best IoU is :%.4f, Best epoch is :%d' % (opt.model_name, new_iou, best_iou, best_epoch))

    return train_loss, new_iou


def train(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, iters, opt, epoch, scaler, log_file=None):
    """通用单输出分割模型训练（UNet/DeepLab/PSPNet/HRNet/……）"""
    global best_iou
    global best_epoch

    epoch_loss = 0.0
    length = 0
    net.train(True)

    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda(non_blocking=True)
        Y = mask.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=AMP_DTYPE):
            output = net(X)                       # [N,C,H,W]
            loss = criterion(output, Y.long())

        scaler.scale(loss).backward()
        clip_gradient(optimizer, opt.clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += float(loss.detach().item())
        length += 1

        with torch.no_grad():
            target_np, pred_np = _to_numpy_mask(output, Y)
            Eva_train.add_batch(target_np, pred_np)

        del output, loss, X, Y
        if i % 50 == 0:
            torch.cuda.empty_cache()

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    Precision = Eva_train.Precision()
    Recall = Eva_train.Recall()
    train_loss = epoch_loss / max(1, length)

    train_msg = 'Epoch [%d/%d], Loss: %.4f, [Training] IOU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f' % (epoch, opt.epoch, train_loss, IOU[1], F1, Precision, Recall)
    log_and_print(train_msg, log_file)
    log_and_print("开始验证...", log_file)

    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda(non_blocking=True)
            Y = mask.cuda(non_blocking=True)
            with autocast(dtype=AMP_DTYPE):
                output = net(X)
            target_np, pred_np = _to_numpy_mask(output, Y)
            Eva_val.add_batch(target_np, pred_np)
            del output, X, Y

    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()
    Precision = Eva_val.Precision()
    Recall = Eva_val.Recall()

    val_msg = 'Epoch [%d/%d], [Validating] IOU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f' % (epoch, opt.epoch, IOU[1], F1, Precision, Recall)
    log_and_print(val_msg, log_file)

    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        best_msg = '🎉 新的最佳模型! %s Model IoU: %.4f; F1: %.4f; Precision: %.4f; Recall: %.4f; Best epoch: %d' % (model_save_name, IOU[1], F1, Precision, Recall, best_epoch)
        log_and_print(best_msg, log_file)
        model_path = os.path.join(opt.save_path, f'{model_save_name}_{opt.data_name}_best.pth')
        torch.save(best_net, model_path)
        save_msg = f'模型已保存到: {model_path}'
        log_and_print(save_msg, log_file)
    else:
        current_msg = '%s Model IoU: %.4f; Best IoU: %.4f (epoch %d)' % (model_save_name, new_iou, best_iou, best_epoch)
        log_and_print(current_msg, log_file)

    return train_loss, new_iou


def train_DBBANet_CTRCL(train_loader, val_loader, net, Eva_train, Eva_val, ce_criterion, dice_criterion, optimizer, iters, opt, epoch, scaler, log_file=None):
    """为 DBBANet_CTRCL 定制的训练循环"""
    global best_iou
    global best_epoch

    epoch_loss = 0.0
    length = 0
    net.train(True)

    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda(non_blocking=True)
        Y = mask.cuda(non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=AMP_DTYPE):
            outputs = net(X, Y, training=True)

            # === 多尺度监督 ===
            # 原始分辨率损失 (CE + Dice)
            loss_seg_cnn_full = ce_criterion(outputs['cnn_output'], Y) + dice_criterion(outputs['cnn_output'], Y, softmax=True)
            loss_seg_trans_full = ce_criterion(outputs['trans_output'], Y) + dice_criterion(outputs['trans_output'], Y, softmax=True)
            loss_seg_final_full = ce_criterion(outputs['final_output'], Y) + dice_criterion(outputs['final_output'], Y, softmax=True)

            # 1/2分辨率监督
            Y_half = F.interpolate(Y.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).long()
            cnn_half = F.interpolate(outputs['cnn_output'], scale_factor=0.5, mode='bilinear', align_corners=False)
            trans_half = F.interpolate(outputs['trans_output'], scale_factor=0.5, mode='bilinear', align_corners=False)
            final_half = F.interpolate(outputs['final_output'], scale_factor=0.5, mode='bilinear', align_corners=False)

            loss_seg_cnn_half = ce_criterion(cnn_half, Y_half) + dice_criterion(cnn_half, Y_half, softmax=True)
            loss_seg_trans_half = ce_criterion(trans_half, Y_half) + dice_criterion(trans_half, Y_half, softmax=True)
            loss_seg_final_half = ce_criterion(final_half, Y_half) + dice_criterion(final_half, Y_half, softmax=True)

            # 1/4分辨率监督
            Y_quarter = F.interpolate(Y.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
            cnn_quarter = F.interpolate(outputs['cnn_output'], scale_factor=0.25, mode='bilinear', align_corners=False)
            trans_quarter = F.interpolate(outputs['trans_output'], scale_factor=0.25, mode='bilinear', align_corners=False)
            final_quarter = F.interpolate(outputs['final_output'], scale_factor=0.25, mode='bilinear', align_corners=False)

            loss_seg_cnn_quarter = ce_criterion(cnn_quarter, Y_quarter) + dice_criterion(cnn_quarter, Y_quarter, softmax=True)
            loss_seg_trans_quarter = ce_criterion(trans_quarter, Y_quarter) + dice_criterion(trans_quarter, Y_quarter, softmax=True)
            loss_seg_final_quarter = ce_criterion(final_quarter, Y_quarter) + dice_criterion(final_quarter, Y_quarter, softmax=True)

            # 多尺度损失加权组合
            loss_seg_cnn = loss_seg_cnn_full + 0.5 * loss_seg_cnn_half + 0.25 * loss_seg_cnn_quarter
            loss_seg_trans = loss_seg_trans_full + 0.5 * loss_seg_trans_half + 0.25 * loss_seg_trans_quarter
            loss_seg_final = loss_seg_final_full + 0.5 * loss_seg_final_half + 0.25 * loss_seg_final_quarter

            # 从标签中提取真实边缘
            Y_edge = kornia.filters.sobel(Y.unsqueeze(1).float()).squeeze(1)
            Y_edge = (Y_edge > 0).float() # 二值化边缘
            loss_edge = F.binary_cross_entropy_with_logits(outputs['edge_output'].squeeze(1), Y_edge)

            # 计算总损失
            total_loss_cnn = loss_seg_cnn + 3 * outputs['rlcl_loss_cnn'] + outputs['cfcl_encoder_loss_cnn'] + 2 * outputs['cfcl_decoder_loss_cnn']
            total_loss_trans = loss_seg_trans + 3 * outputs['rlcl_loss_trans'] + outputs['cfcl_encoder_loss_trans'] + 2 * outputs['cfcl_decoder_loss_trans']

            # 将边缘损失添加到总损失中，使用更强的边缘权重
            total_loss = total_loss_cnn + total_loss_trans + loss_seg_final + 1.0 * loss_edge

        scaler.scale(total_loss).backward()
        clip_gradient(optimizer, opt.clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += float(total_loss.detach().item())
        length += 1

        with torch.no_grad():
            # 评估最终的输出
            target_np, pred_np = _to_numpy_mask(outputs['final_output'], Y)
            Eva_train.add_batch(target_np, pred_np)

        del outputs, total_loss, X, Y
        if i % 50 == 0:
            torch.cuda.empty_cache()

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    Precision = Eva_train.Precision()
    Recall = Eva_train.Recall()
    train_loss = epoch_loss / max(1, length)

    train_msg = 'Epoch [%d/%d], Loss: %.4f, [Training] IOU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f' % (epoch, opt.epoch, train_loss, IOU[1], F1, Precision, Recall)
    log_and_print(train_msg, log_file)
    log_and_print("开始验证...", log_file)

    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda(non_blocking=True)
            Y = mask.cuda(non_blocking=True).long()
            with autocast(dtype=AMP_DTYPE):
                output = net(X, training=False)
            target_np, pred_np = _to_numpy_mask(output, Y)
            Eva_val.add_batch(target_np, pred_np)
            del output, X, Y

    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()
    Precision = Eva_val.Precision()
    Recall = Eva_val.Recall()

    val_msg = 'Epoch [%d/%d], [Validating] IOU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f' % (epoch, opt.epoch, IOU[1], F1, Precision, Recall)
    log_and_print(val_msg, log_file)

    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        best_msg = '🎉 新的最佳模型! %s Model IoU: %.4f; F1: %.4f; Precision: %.4f; Recall: %.4f; Best epoch: %d' % (model_save_name, IOU[1], F1, Precision, Recall, best_epoch)
        log_and_print(best_msg, log_file)
        model_path = os.path.join(opt.save_path, f'{model_save_name}_{opt.data_name}_best.pth')
        torch.save(best_net, model_path)
        save_msg = f'模型已保存到: {model_path}'
        log_and_print(save_msg, log_file)
    else:
        current_msg = '%s Model IoU: %.4f; Best IoU: %.4f (epoch %d)' % (model_save_name, new_iou, best_iou, best_epoch)
        log_and_print(current_msg, log_file)

    return train_loss, new_iou


def train_DBBANet_Enhanced(train_loader, val_loader, net, Eva_train, Eva_val, optimizer, iters, opt, epoch, scaler, log_file=None):
    """为 DBBANet_Enhanced 定制的训练循环 - 使用内置的组合损失函数"""
    global best_iou
    global best_epoch

    epoch_loss = 0.0
    length = 0
    net.train(True)

    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda(non_blocking=True)
        Y = mask.cuda(non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=AMP_DTYPE):
            outputs = net(X, Y, training=True)

            # 使用模型内置的组合损失函数，无需额外计算
            # 模型已经在forward中计算了 loss_seg_cnn, loss_seg_trans, loss_seg_final

            # 从标签中提取真实边缘用于边缘监督
            Y_edge = kornia.filters.sobel(Y.unsqueeze(1).float()).squeeze(1)
            Y_edge = (Y_edge > 0).float()  # 二值化边缘
            loss_edge = F.binary_cross_entropy_with_logits(outputs['edge_output'].squeeze(1), Y_edge)

            # 计算总损失 - 平衡各损失权重
            total_loss_cnn = outputs['loss_seg_cnn'] + 0.5 * outputs['rlcl_loss_cnn'] + 0.3 * outputs['cfcl_encoder_loss_cnn'] + 0.3 * outputs['cfcl_decoder_loss_cnn']
            total_loss_trans = outputs['loss_seg_trans'] + 0.5 * outputs['rlcl_loss_trans'] + 0.3 * outputs['cfcl_encoder_loss_trans'] + 0.3 * outputs['cfcl_decoder_loss_trans']

            # 总损失包含边缘损失 - 降低边缘损失权重
            total_loss = 0.4 * total_loss_cnn + 0.4 * total_loss_trans + outputs['loss_seg_final'] + 0.2 * loss_edge

        scaler.scale(total_loss).backward()
        clip_gradient(optimizer, opt.clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += float(total_loss.detach().item())
        length += 1

        with torch.no_grad():
            target_np, pred_np = _to_numpy_mask(outputs['final_output'], Y)
            Eva_train.add_batch(target_np, pred_np)

        del outputs, total_loss, X, Y
        if i % 50 == 0:
            torch.cuda.empty_cache()

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    Precision = Eva_train.Precision()
    Recall = Eva_train.Recall()
    train_loss = epoch_loss / max(1, length)

    train_msg = 'Epoch [%d/%d], Loss: %.4f, [Training] IOU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f' % (epoch, opt.epoch, train_loss, IOU[1], F1, Precision, Recall)
    log_and_print(train_msg, log_file)
    log_and_print("开始验证...", log_file)

    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda(non_blocking=True)
            Y = mask.cuda(non_blocking=True)
            with autocast(dtype=AMP_DTYPE):
                output = net(X, training=False)  # 推理模式，只返回最终输出
            target_np, pred_np = _to_numpy_mask(output, Y)
            Eva_val.add_batch(target_np, pred_np)
            del output, X, Y

    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()
    Precision = Eva_val.Precision()
    Recall = Eva_val.Recall()

    val_msg = '[Validation] IOU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f' % (IOU[1], F1, Precision, Recall)
    log_and_print(val_msg, log_file)

    val_iou = IOU[1]
    if val_iou > best_iou:
        best_iou = val_iou
        best_epoch = epoch
        # 确保保存目录存在
        os.makedirs(opt.save_path, exist_ok=True)
        model_path = os.path.join(opt.save_path, f'{opt.model_name}_{opt.data_name}_best.pth')
        torch.save(net.state_dict(), model_path)
        best_msg = f'✅ 新的最佳模型已保存! IOU: {best_iou:.4f} (Epoch {best_epoch})\n    保存路径: {model_path}'
        log_and_print(best_msg, log_file)

    current_msg = f'当前最佳: IOU {best_iou:.4f} (Epoch {best_epoch}), 本轮: IOU {val_iou:.4f} (Epoch {epoch})'
    log_and_print(current_msg, log_file)

    return train_loss, val_iou


class hyper_parameters:
    def __init__(self):
        self.blocks = ['BOTTLENECK','BASIC','BASIC','BASIC']
        self.num_modules = [1, 1, 4, 3]
        self.num_branches = [1, 2, 3, 4]
        self.num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
        self.num_channels = [[64], [48, 96], [48, 96, 192], [48, 96, 192, 384]]
        self.fuse_method = ['sum', 'sum', 'sum', 'sum']


if __name__ == '__main__':
    import argparse
    torch.manual_seed(3407)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--model_name', type=str, default='DBBANet_CTRCL', help='the model')

    parser.add_argument('--data_name', type=str, default='FarmSeg', help='the test rgb images root')
    parser.add_argument('--segclass', type=int, default=2, help='')
    parser.add_argument('--save_path', type=str, default=r'/root/lanyun-fs/Code2/FarmSeg2/DBBANet_CTRCL')
    parser.add_argument('--val', type=str, default='test')
    parser.add_argument('--rename', type=str, default=None)
    opt = parser.parse_args()

    # 推荐：显存吃紧时先把 batchsize/trainsize 降一点再试
    # opt.batchsize = min(opt.batchsize, 2)
    # opt.trainsize = min(opt.trainsize, 384)

    # 设置日志文件路径
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(opt.save_path, f"{opt.model_name}_{opt.data_name}_{timestamp}.log")

    # 根据数据集设置调色板
    if opt.data_name == 'FarmSeg-VL':
        # FarmSeg-VL数据集专用调色板：黑色背景 + 亮红色农田
        palette = [[0,0,0],[255,0,0]]
    else:
        # 原始FarmSeg数据集调色板：黑色背景 + 红色 + 绿色农田
        palette = [[0,0,0],[128,0,0],[0,128,0]]

    if opt.data_name == 'FarmSeg':
        opt.train_root = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg/train/img/'
        opt.train_gt = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg/train/label/'
        if opt.val =='val':
            opt.val_root = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg/val/img/'
            opt.val_gt = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg/val/label/'
        elif opt.val =='test':
            opt.val_root = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg/test/img/'
            opt.val_gt = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg/test/label/'
    
    elif opt.data_name == 'FarmSeg-VL':
        opt.train_root = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg-VL/train/img/'
        opt.train_gt = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg-VL/train/label/'
        if opt.val =='val':
            opt.val_root = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg-VL/val/img/'
            opt.val_gt = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg-VL/val/label/'
        elif opt.val =='test':
            opt.val_root = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg-VL/test/img/'
            opt.val_gt = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg-VL/test/label/'

    # set the device for training（注意：这在 import torch 之后，所以只影响后续 .cuda() 的默认设备）
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    if opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    if opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    if opt.gpu_id == '4':
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        print('USE GPU 4')
    if opt.gpu_id == '5':
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        print('USE GPU 5')
    elif opt.gpu_id == '6':
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        print('USE GPU 6')
    elif opt.gpu_id == '7':
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        print('USE GPU 7')

    train_loader = dataloader.get_loader(opt.train_root, opt.train_gt, palette, opt.batchsize, opt.trainsize, mode='train', num_workers=16, shuffle=True, pin_memory=True)
    val_loader = dataloader.get_loader(opt.val_root, opt.val_gt, palette, opt.batchsize, opt.trainsize, mode='val', num_workers=16, shuffle=False, pin_memory=True)

    # --- 构建模型 ---
    if opt.model_name =='UNet':
        net = UNet.UNetWithResnet50Encoder(n_classes=opt.segclass).cuda()
    elif opt.model_name =='DeeplabV3+':
        net = deeplabv3.DeepLab(num_classes=opt.segclass).cuda()
    elif opt.model_name == 'PSPNet':
        net = Pspnet(num_classes=opt.segclass).cuda()
    elif opt.model_name =='HRNet':
        hp = hyper_parameters()
        net = HRNet.HighResolutionNet(hp.blocks, hp.num_channels, hp.num_modules, hp.num_branches, hp.num_blocks,
                                        hp.fuse_method,opt.segclass).cuda()
    elif opt.model_name == 'ABCNet':
        net = ABCNet.ABCNet(3, opt.segclass).cuda()
    elif opt.model_name == 'CMTFNet':
        net = CMTFNet.CMTFNet(num_classes= opt.segclass).cuda()
    elif opt.model_name == 'MCCANet':
        net = MCCA.MCCA(num_class = opt.segclass).cuda()
    elif opt.model_name =='CGNet':
        net = CGNet.Context_Guided_Network(classes = opt.segclass).cuda()
    elif opt.model_name == 'DenseASPP':
        net = DenseASPP.DenseASPP(n_class = opt.segclass).cuda()
    elif opt.model_name =='ENet':
        net =ENet.ENet(n_classes = opt.segclass).cuda()
    elif opt.model_name =='SegNet':
        net = SegNet.SegNet(classes=opt.segclass).cuda()
    elif opt.model_name == 'BuildFormer':
        net = BuildFormer.BuildFormerSegDP(num_classes=opt.segclass).cuda()
    elif opt.model_name =='UANet':
        net = UANet_final_Res50(channel=32, num_classes=opt.segclass).cuda()
    elif opt.model_name =='DSNet':
        net = DSNet.DSNetLocalGuide(classes = opt.segclass).cuda()
    elif opt.model_name =='UNetFormer':
        net = UNetFormer(num_classes = opt.segclass).cuda()
    elif opt.model_name =='DBBANet':
        net = DBBANet(num_class=opt.segclass).cuda()
        print('🚀 DBBANet initialized with ResNet50 backbone + GBC + Frequency Enhancement (默认启用)')
    elif opt.model_name == 'DBBANet_CTRCL':
        net = DBBANet_CTRCL(num_class=opt.segclass).cuda()
        print('🚀 DBBANet_CTRCL initialized with CNN and Transformer backbones')
    elif opt.model_name == 'DBBANet_Enhanced':
        net = DBBANet_Enhanced(num_class=opt.segclass).cuda()
        print('🚀 DBBANet_Enhanced initialized with Enhanced Edge Decoder + Combined Loss')
    if opt.load is not None:
        net.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    if opt.rename is not None:
        opt.model_name = opt.rename

    # 设置模型保存名称
    # --- 设置优化器 ---
    if opt.model_name in ['DBBANet_CTRCL', 'DBBANet_Enhanced']:
        # 为 CTRCL 和 Enhanced 模型设置差分学习率
        cnn_backbone_params = list(filter(lambda p: p.requires_grad, net.cnn_backbone.parameters()))
        transformer_backbone_params = list(filter(lambda p: p.requires_grad, net.transformer_backbone.parameters()))
        decoder_params = [p for name, p in net.named_parameters() if 'backbone' not in name and p.requires_grad]

        optimizer = torch.optim.Adam([
            {'params': cnn_backbone_params, 'lr': opt.lr * 0.1}, # CNN主干使用较小的学习率
            {'params': transformer_backbone_params, 'lr': opt.lr}, # Transformer主干使用基准学习率
            {'params': decoder_params, 'lr': opt.lr} # 其他部分使用基准学习率
        ], lr=opt.lr, weight_decay=1e-5)
        print(f"✅ 已为 {opt.model_name} 设置差分学习率 (CNN backbone: 0.1*lr, Transformer & Decoders: lr).")
    else:
        # 其他模型使用标准优化器
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, weight_decay=1e-5)
    # ---
    # ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=opt.lr * 0.01)
    print(f"✅ 已设置 CosineAnnealingLR 学习率调度器 (T_max={opt.epoch}, eta_min={opt.lr * 0.01}).")

    model_save_name = opt.model_name

    opt.vis_path = './Vis/'+ model_save_name + '/' +opt.data_name +'/'
    if not os.path.exists(opt.vis_path):
        os.makedirs(opt.vis_path)

    edge_criterion = cross_entropy_loss_RCF
    criterion = nn.CrossEntropyLoss().cuda() # 保留原名，兼容旧模型
    dice_criterion = DiceLoss(n_classes=opt.segclass).cuda() # 为新模型准备
    best_iou = 0
    best_epoch = 0
    loss_train = []
    iou_val = []
    Epoch_all = []

    # AMP GradScaler
    scaler = GradScaler()

    log_and_print("="*50, log_file)
    log_and_print("开始训练!", log_file)
    log_and_print(f"训练模型: {opt.model_name}", log_file)
    log_and_print(f"数据集: {opt.data_name}", log_file)
    log_and_print(f"分割类别数: {opt.segclass}", log_file)
    log_and_print(f"验证数据: {opt.val}", log_file)
    log_and_print("="*50, log_file)

    for epoch in range(1, opt.epoch+1):
        Eva_train = Evaluator(opt.segclass)
        Eva_val = Evaluator(opt.segclass)
        # 从优化器直接获取当前学习率用于打印
        current_lr = optimizer.param_groups[0]['lr']
        log_and_print(f'Epoch {epoch}/{opt.epoch} - Learning rate: {current_lr:.6f}', log_file)

        if opt.model_name ==  'ABCNet':
            train_loss, val_iou = train_ABC(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch, scaler, log_file)
        elif opt.model_name ==  'UANet':
            train_loss, val_iou = train_UANet(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch, scaler, log_file)
        elif opt.model_name ==  'DSNet':
            train_loss, val_iou = train_DSNet(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch, scaler, log_file)
        elif opt.model_name ==  'UNetFormer':
            train_loss, val_iou = train_UNetFormer(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch, scaler, log_file)
        elif opt.model_name == 'DBBANet_CTRCL':
            train_loss, val_iou = train_DBBANet_CTRCL(train_loader, val_loader, net, Eva_train, Eva_val, criterion, dice_criterion, optimizer, 0, opt, epoch, scaler, log_file)
        elif opt.model_name == 'DBBANet_Enhanced':
            train_loss, val_iou = train_DBBANet_Enhanced(train_loader, val_loader, net, Eva_train, Eva_val, optimizer, 0, opt, epoch, scaler, log_file)
        else:
            train_loss, val_iou =  train(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch, scaler, log_file)

        Epoch_all.append(epoch)
        loss_train.append(train_loss)
        iou_val.append(val_iou)

        # 更新学习率调度器
        scheduler.step()

    # 训练完成总结
    log_and_print("="*50, log_file)
    log_and_print("训练完成!", log_file)
    log_and_print(f"最佳IoU: {best_iou:.4f} (第 {best_epoch} 轮)", log_file)
    log_and_print(f"最终训练损失: {loss_train[-1]:.4f}", log_file)
    log_and_print(f"最终验证IoU: {iou_val[-1]:.4f}", log_file)
    log_and_print("="*50, log_file)

    # --- 可视化 ---
    plt.plot(Epoch_all, loss_train, 'r--', label='train_loss')
    plt.xlabel('epoch'); plt.ylabel('train_loss')
    plt.savefig(opt.vis_path + 'train_loss.png', bbox_inches='tight')

    plt.clf()
    plt.plot(Epoch_all, iou_val, 'r--', label='val_iou')
    plt.xlabel('epoch'); plt.ylabel('val_iou')
    plt.savefig(opt.vis_path + 'val_iou.png', bbox_inches='tight')

    with open(opt.vis_path + 'train_loss.txt', 'w') as f:
        for v in loss_train:
            f.write(str(v) + '\n')
    with open(opt.vis_path + 'val_iou.txt', 'w') as f:
        for v in iou_val:
            f.write(str(v) + '\n')

    log_and_print("训练流程全部完成! 🎉", log_file)


