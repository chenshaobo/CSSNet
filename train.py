r""" CSSNet training (validation) code """
import argparse
import torch.optim as optim
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from torch.optim import AdamW  # 改为使用AdamW优化器
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR  # 添加余弦退火调度器

from model.CSSNet import CSSNet
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from model.base.srm import apply_srm_to_batch

def train(epoch, model, dataloader, optimizer, training, dataset, scaler=None):
    r""" Train CSSNet """
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        if training and scaler is not None:
            # 使用自动混合精度
            with torch.amp.autocast('cuda'):
                outputs = model(batch['query_img'],
                               batch['support_imgs'].squeeze(1),
                               batch['support_masks'].squeeze(1),
                               batch['history_mask'])

                logit_mask = outputs['logit_mask']
                pred_softmax = F.softmax(logit_mask, dim=1).detach().cpu()
                pred_mask = F.interpolate(logit_mask, size=batch['query_img'].size()[-2:],
                                          mode='bilinear', align_corners=True).argmax(dim=1)

                # Compute loss
                loss = model.module.compute_objective(
                    logit_mask,
                    batch['query_mask'],
                    outputs
                )

            # 更新历史掩码
            for j in range(batch['query_img'].shape[0]):
                sub_index = batch['idx'][j]
                dataset.history_mask_list[sub_index] = pred_softmax[j]

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # 添加梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch['query_img'],
                           batch['support_imgs'].squeeze(1),
                           batch['support_masks'].squeeze(1),
                           batch['history_mask'])

            logit_mask = outputs['logit_mask']
            pred_softmax = F.softmax(logit_mask, dim=1).detach().cpu()
            pred_mask = F.interpolate(logit_mask, size=batch['query_img'].size()[-2:],
                                      mode='bilinear', align_corners=True).argmax(dim=1)

            # Update history mask
            for j in range(batch['query_img'].shape[0]):
                sub_index = batch['idx'][j]
                dataset.history_mask_list[sub_index] = pred_softmax[j]

            # Compute loss
            loss = model.module.compute_objective(
                logit_mask,
                batch['query_mask'],
                outputs
            )

            if training:
                optimizer.zero_grad()
                loss.backward()

                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

        # Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    return avg_loss, miou, fb_iou


def validate(epoch, model, dataloader, dataset, scaler=None):
    """ Validation using the t_10.py testing approach """
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        logit_mask_agg = 0
        logit_mask_avg = 0

        # Multiple shot prediction (assuming nshot=1 for validation)
        s_idx = 0
        # Get model outputs
        with torch.amp.autocast('cuda') if scaler is not None else torch.no_grad():
            outputs = model(
                batch['query_img'],
                batch['support_imgs'][:, s_idx],
                batch['support_masks'][:, s_idx],
                batch['history_mask']
            )
        logit_mask = outputs['logit_mask']

        # Handle original image size (always False in training)
        logit_mask = F.interpolate(
            logit_mask,
            batch['support_imgs'].size()[-2:],
            mode='bilinear',
            align_corners=True
        )

        # Compute loss
        # 在验证阶段禁用混合精度以避免类型不匹配问题
        with torch.amp.autocast('cuda', enabled=False):
            # 确保所有张量都是float类型
            float_outputs = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and scaler is not None:
                    float_outputs[k] = v.float()
                else:
                    float_outputs[k] = v

            loss = model.module.compute_objective(
                logit_mask.float() if scaler is not None else logit_mask,
                batch['query_mask'],
                float_outputs
            )

        # Accumulate predictions
        logit_mask_avg += F.softmax(logit_mask, dim=1).detach().cpu()
        logit_mask_agg += logit_mask.argmax(dim=1).clone()

        # Calculate average probability
        avg_prob = logit_mask_avg / 1.0  # [B, 2, H, W]

        # Original prediction method (no SRM in validation)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote

        threshold = 0.5
        pred_mask[pred_mask < threshold] = 0
        pred_mask[pred_mask >= threshold] = 1

        # Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask.clone(), batch
        )
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSSNet Pytorch Implementation')
    #parser.add_argument('--benchmark', type=str, default='fss', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--datapath', type=str, default=r'data\VOCdevkit')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--niter', type=int, default=300)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--loadpath', type=str, default='')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision training')
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = CSSNet(args.backbone, False)

    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    else:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 创建GradScaler用于混合精度训练
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None

    # 使用余弦退火学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=300,  # 最大迭代次数
        eta_min=1e-6       # 最小学习率
    )

    Evaluator.initialize()
    current_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.loadpath)
        current_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['net'])
        # 检查优化器状态是否兼容
        if 'optimizer' in checkpoint and isinstance(checkpoint['optimizer'], dict):
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                Logger.info("Optimizer state loaded successfully")
            except:
                Logger.warning("Failed to load optimizer state. Using new optimizer.")
        # 如果检查点中包含scaler状态，则加载
        if args.use_amp and 'scaler' in checkpoint and scaler is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler'])
                Logger.info("Scaler state loaded successfully")
            except:
                Logger.warning("Failed to load scaler state. Using new scaler.")
        model.train()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn, dataset_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val, dataset_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    # Train CSSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(current_epoch, args.niter):
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, dataset=dataset_trn, training=True, scaler=scaler)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = validate(epoch, model, dataloader_val, dataset_val, scaler=scaler)
            Logger.info(
                f"Validated model @Epoch {epoch} mIoU: {val_miou:.2f}   FB-IoU: {val_fb_iou:.2f}   Loss: {val_loss:.4f}")
        # 更新学习率 (使用余弦退火)
        scheduler.step()

        # 或者使用ReduceLROnPlateau
        # scheduler_plateau.step(val_miou)

        # Save best model and re-validate
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou, optimizer)
            # 如果使用混合精度，同时保存scaler状态
            if args.use_amp and scaler is not None:
                checkpoint = {
                    'epoch': epoch,
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()  # 保存scaler状态
                }
                #torch.save(checkpoint, os.path.join(Logger.logpath, 'best_model.pth'))

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        Logger.info(f'Epoch {epoch}: LR = {current_lr:.6f}')
        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.add_scalar('data/learning_rate', current_lr, epoch)
        Logger.tbd_writer.flush()

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
    Logger.info(f'Best Validation mIoU: {best_val_miou:.4f}')