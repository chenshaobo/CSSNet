r""" Hypercorrelation Squeeze Network with CSFM """
from functools import reduce
from operator import add
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet, vgg
from .base.feature import extract_feat_vgg, extract_feat_res, extract_feat_chossed, extract_feat_vgg_dense
from .base.correlation import Correlation
from .learner import HPNLearner
from .base.CRM import GCCG
from .base.FEM import FEM
from .CSFM import CSFM

class CSSNet(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(CSSNet, self).__init__()
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        import os
        self.csfm_enabled = os.environ.get('ABL_CSFM', '1') == '1'
        self.intervention_enabled = os.environ.get('ABL_INTERV', '1') == '1'
        self.adversarial_enabled = os.environ.get('ABL_ADV', '1') == '1'
        self.sparsity_enabled = os.environ.get('ABL_SPARSITY', '1') == '1'
        self.aux_enabled = os.environ.get('ABL_AUX', '1') == '1'

        # CSFM configuration
        if backbone == 'resnet50':
            self.csfm_modules = nn.ModuleList([
                CSFM(in_channels=ch, reduction_ratio=16) for ch in [512, 1024, 2048]
            ])
            last_ch = 2048
        elif backbone == 'vgg16':
            self.csfm_modules = nn.ModuleList([
                CSFM(in_channels=ch, reduction_ratio=8) for ch in [256, 512, 512]
            ])
            last_ch = 512
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        # Loss weights
        self.reg_weight = 0.1
        self.contrast_weight = 0.5
        self.adversarial_weight = 0.3

        # Auxiliary heads
        self.aux_head_con = nn.Sequential(
            nn.Conv2d(last_ch, last_ch//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(last_ch//4, 2, kernel_size=1)
        )
        self.aux_head_ad = nn.Sequential(
            nn.Conv2d(last_ch, last_ch//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(last_ch//4, 2, kernel_size=1)
        )

        # Backbone initialization
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        else:
            raise ValueError(f'Unavailable backbone: {backbone}')

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()

        # Learner modules
        if self.backbone_type == 'resnet50':
            self.hpn_learner = HPNLearner([4, 7, 5])
        if self.backbone_type == 'vgg16':
            self.hpn_learner = HPNLearner([2, 4, 4])

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.ss = GCCG(3, output_channel=64)

        dims = [512, 1024, 2048]
        vgg_dims = [256, 512, 512]

        # FEM initialization
        if torch.cuda.device_count() == 1:
            if self.backbone_type == 'resnet50':
                self.fem = [FEM(dims[i]).cuda() for i in range(3)]
            elif self.backbone_type == 'vgg16':
                self.fem = [FEM(vgg_dims[i]).cuda() for i in range(3)]
        else:
            self.fem_1 = FEM(512).cuda()
            self.fem_2 = FEM(1024).cuda()
            self.fem_3 = FEM(2048).cuda()

    def forward(self, query_img, support_img, support_mask, history_mask_pred):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone)
            support_feats = self.extract_feats(support_img, self.backbone)

            # CSFM processing
            csfm_outputs = []
            for i, (q_feat, s_feat) in enumerate(zip(query_feats, support_feats)):
                csfm_out = self.csfm_modules[i](q_feat, s_feat)
                csfm_outputs.append(csfm_out)
                # Update features with enhanced versions
                query_feats[i] = csfm_out['enhanced_query']
                support_feats[i] = csfm_out['enhanced_support']

            # Feature extraction for correlation
            if self.backbone_type == 'resnet50':
                query_feats_dense = extract_feat_chossed(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats_dense = extract_feat_chossed(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats = self.mask_feature(support_feats, support_mask.clone())
                corr_dense = Correlation.multilayer_correlation_dense(query_feats_dense, support_feats_dense, self.stack_ids)
            elif self.backbone_type == 'vgg16':
                query_feats_dense = extract_feat_vgg_dense(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats_dense = extract_feat_vgg_dense(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats[2] = F.interpolate(support_feats[2], (13, 13), mode='bilinear', align_corners=True)
                query_feats[2] = F.interpolate(query_feats[2], (13, 13), mode='bilinear', align_corners=True)
                support_feats_dense[-1] = F.interpolate(support_feats_dense[-1], (13, 13), mode='bilinear', align_corners=True)
                query_feats_dense[-1] = F.interpolate(query_feats_dense[-1], (13, 13), mode='bilinear', align_corners=True)
                support_feats = self.mask_feature(support_feats, support_mask.clone())
                corr_dense = Correlation.multilayer_correlation_dense(query_feats_dense, support_feats_dense, self.stack_ids)

        # Feature enhancement
        if torch.cuda.device_count() == 1:
            for i in range(3):
                query_feats[i], support_feats[i] = self.fem[i](query_feats[i], support_feats[i])
        else:
            query_feats[0], support_feats[0] = self.fem_1(query_feats[0], support_feats[0])
            query_feats[1], support_feats[1] = self.fem_2(query_feats[1], support_feats[1])
            query_feats[2], support_feats[2] = self.fem_3(query_feats[2], support_feats[2])

        # Global context correlation
        similarity_s, similarity_q = [], []
        for i in range(len(query_feats)):
            similarity_q.append(self.ss(query_feats[i]))
            similarity_s.append(self.ss(support_feats[i]))
        corr_self_simi = Correlation.multilayer_correlation(similarity_q, similarity_s, self.stack_ids)

        # Concatenate correlations
        for i in range(len(corr_dense)):
            corr_dense[i] = torch.cat([corr_dense[i], corr_self_simi[i]], dim=1)

        logit_mask = self.hpn_learner(corr_dense, history_mask_pred)

        # Prepare outputs
        outputs = {
            'logit_mask': logit_mask,
            'csfm_outputs': csfm_outputs,
            'query_feats': query_feats
        }
        return outputs

    def compute_objective(self, logit_mask, gt_mask, outputs):
        # Segmentation loss
        seg_loss = self.cross_entropy_loss(
            F.interpolate(logit_mask, gt_mask.size()[-2:], mode='bilinear', align_corners=True),
            gt_mask.long()
        )

        # CSFM regularization and contrastive losses
        reg_loss = 0
        contrast_loss = 0
        adversarial_loss = 0

        for ppi_out in outputs['csfm_outputs']:
            if self.sparsity_enabled:
                reg_loss += ppi_out['reg_loss']
            if self.intervention_enabled:
                contrast_loss += ppi_out['contrast_loss']
            if self.adversarial_enabled:
                adversarial_loss += ppi_out['adversarial_loss']

        # Auxiliary losses
        b, c, h, w = outputs['query_feats'][-1].shape
        rand_idx = torch.randperm(b)
        adv_feats = outputs['query_feats'][-1][rand_idx]

        # 使用autocast确保在AMP训练时类型匹配
        with torch.cuda.amp.autocast(enabled=False):
            aux_con = self.aux_head_con(outputs['csfm_outputs'][-1]['intervention'].float())
            aux_ad = self.aux_head_ad(adv_feats.float())

            if gt_mask.dim() == 3:  # 如果缺少通道维度
                gt_mask = gt_mask.unsqueeze(1)  # 添加通道维度
            # 为aux_con和aux_ad分别创建尺寸匹配的标签
            aux_gt_con = F.interpolate(gt_mask.float(), aux_con.shape[-2:], mode='nearest').long().squeeze(1)
            aux_gt_ad = F.interpolate(gt_mask.float(), aux_ad.shape[-2:], mode='nearest').long().squeeze(1)
            aux_con_loss = self.cross_entropy_loss(aux_con, torch.zeros_like(aux_gt_con))
            aux_ad_loss = self.cross_entropy_loss(aux_ad, aux_gt_ad)

        # Total loss
        #打印所有损失
        #print('seg_loss', seg_loss.item(), 'reg_loss', reg_loss.item(), 'contrast_loss', contrast_loss.item(), 'adversarial_loss', adversarial_loss, 'aux_con_loss', aux_con_loss, 'aux_ad_loss', aux_ad_loss)
        total_loss = (
            seg_loss +
            self.reg_weight * reg_loss +
            self.contrast_weight * (contrast_loss + aux_con_loss) +
            self.adversarial_weight * (adversarial_loss + aux_ad_loss)
        )
        return total_loss

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def train_mode(self):
        self.train()
        self.backbone.eval()