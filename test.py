r""" CSSNet Testing Code (Updated for New Model) """
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.CSSNet import CSSNet
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import os
import numpy as np
from model.base.srm import apply_srm_to_batch

def t(model, dataloader, nshot, dataset, use_srm=True):
    """ Test CSSNet with SRM enhancement """
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        logit_mask_agg = 0
        logit_mask_avg = 0

        # Multiple shot prediction
        for s_idx in range(nshot):
            # Get model outputs (new dictionary interface)
            outputs = model(
                batch['query_img'],
                batch['support_imgs'][:, s_idx],
                batch['support_masks'][:, s_idx],
                batch['history_mask']
            )
            logit_mask = outputs['logit_mask']  # Extract logit_mask from dictionary

            # Handle original image size
            if dataset.use_original_imgsize:
                pred_softmax = F.softmax(logit_mask, dim=1).detach().cpu()
                for j in range(batch['query_img'].shape[0]):
                    sub_index = batch['idx'][j]
                    dataset.history_mask_list[sub_index] = pred_softmax[j]
            else:
                logit_mask = F.interpolate(
                    logit_mask,
                    batch['support_imgs'].size()[-2:],
                    mode='bilinear',
                    align_corners=True
                )

            # Accumulate predictions
            logit_mask_avg += F.softmax(logit_mask, dim=1).detach().cpu()
            if dataset.use_original_imgsize:
                org_qry_imsize = tuple([
                    batch['org_query_imsize'][1].item(),
                    batch['org_query_imsize'][0].item()
                ])
                logit_mask = F.interpolate(
                    logit_mask,
                    org_qry_imsize,
                    mode='bilinear',
                    align_corners=True
                )

            logit_mask_agg += logit_mask.argmax(dim=1).clone()

        # Calculate average probability
        avg_prob = logit_mask_avg / nshot  # [B, 2, H, W]

        # ==== SRM优化处理 ====
        if use_srm and not dataset.use_original_imgsize:
            final_prob = apply_srm_to_batch(batch, avg_prob)
            final_prob = final_prob.to(batch['query_img'].device)  # 确保张量在正确的设备上
            pred_mask = final_prob.argmax(dim=1)  # [B, H, W]
        else:
            # Original prediction method
            bsz = logit_mask_agg.size(0)
            max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
            max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
            max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
            pred_mask = logit_mask_agg.float() / max_vote

            threshold = 0.5
            pred_mask[pred_mask < threshold] = 0
            pred_mask[pred_mask >= threshold] = 1

            # If using original image size, resize prediction
        if dataset.use_original_imgsize:
            org_qry_imsize = tuple([
                batch['org_query_imsize'][1].item(),
                batch['org_query_imsize'][0].item()
            ])
            pred_mask = F.interpolate(
                pred_mask.unsqueeze(1).float() if pred_mask.dim() == 3 else pred_mask.float(),
                size=org_qry_imsize,
                mode='nearest'
            )
            if pred_mask.dim() == 4:
                pred_mask = pred_mask.squeeze(1)

            # 确保pred_mask在正确的设备上
        if pred_mask.device != batch['query_mask'].device:
            pred_mask = pred_mask.to(batch['query_mask'].device)

            # Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask.clone(), batch
        )
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(
                batch['support_imgs'], batch['support_masks'],
                batch['query_img'], batch['query_mask'],
                pred_mask, batch['class_id'], idx, batch['query_name'],
                batch['support_names'],
                area_inter[1].float() / area_union[1].float()
            )

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou


    # Arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSSNet Pytorch Implementation')
    #parser.add_argument('--datapath', type=str, default=r'D:\pycharm\datasets\coco2014')
    parser.add_argument('--datapath', type=str, default=r'D:\pycharm\CSSNet\data\VOCdevkit')
    #parser.add_argument('--datapath', type=str, default=r'D:\pycharm\datasets\FSS-1000')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='logs/RN50_pascal_0.log/best_model.pt')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    parser.add_argument('--visualize', action='store_true',default=False)
    parser.add_argument('--use_original_imgsize', action='store_true', default=False)
    parser.add_argument('--visual_fold_name', type=str, default='pascal_1_srm')
    parser.add_argument('--use_srm', action='store_true', default=True, help='Enable SRM post-processing')
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = CSSNet(args.backbone, args.use_original_imgsize)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    if torch.cuda.device_count() >= 2:
        device_ids = [0, 1]  # 指定使用两张显卡
        model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
        Logger.info(f"Using {torch.cuda.device_count()} GPUs for testing")
    else:
        model.to(device)
        Logger.info("Using single GPU for testing")

    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '':
        raise Exception('Pretrained model not specified.')
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['net'], strict=False)

    # Helper classes initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.visual_fold_name)

    # Dataset initialization
    FSSDataset.initialize(
        img_size=400,
        datapath=args.datapath,
        use_original_imgsize=args.use_original_imgsize
    )
    dataloader_test, dataset_val = FSSDataset.build_dataloader(
        args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot
    )

    # Test CSSNet
    with torch.no_grad():
        test_miou, test_fb_iou = t(model, dataloader_test, args.nshot, dataset_val, use_srm=args.use_srm)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' %
                (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
