import torch
import torch.nn as nn
import torch.nn.functional as F


class CSFM(nn.Module):
    """CSFM module with causal learning design by fzz"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(CSFM, self).__init__()
        self.in_channels = in_channels

        # Saliency mapping
        self.saliency_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Intervention parameters
        self.tau = 0.25  # Temperature for soft masking
        self.omega = 30  # Scaling factor

    def forward(self, query_feat, support_feat):
        # Generate saliency maps
        q_saliency = self.saliency_conv(query_feat)
        s_saliency = self.saliency_conv(support_feat)

        # Soft masking function
        def soft_mask(saliency):
            return 1 / (1 + torch.exp(-self.omega * (saliency - self.tau)))

        # Generate enhanced features
        enhanced_query = query_feat * (1 + q_saliency)
        enhanced_support = support_feat * (1 + s_saliency)

        # Pseudo-interventions
        intervention_query = query_feat * (1 - soft_mask(q_saliency))
        #intervention_support = support_feat * (1 - soft_mask(s_saliency))

        # Adversarial interventions
        batch_size = query_feat.size(0)
        rand_idx = torch.randperm(batch_size)
        adv_query = query_feat * (1 - soft_mask(q_saliency[rand_idx]))
        adv_support = support_feat * (1 - soft_mask(s_saliency[rand_idx]))

        # Contrastive loss
        contrast_loss = F.mse_loss(intervention_query, torch.zeros_like(intervention_query))

        # Adversarial loss
        feat_elements = query_feat.numel() / batch_size  # 每个样本的元素数量
        adversarial_loss = (F.mse_loss(adv_query, query_feat) + F.mse_loss(adv_support, support_feat)) / feat_elements

        # Regularization loss
        reg_loss = (torch.mean(torch.abs(q_saliency)) + torch.mean(torch.abs(s_saliency)))/2

        return {
            'enhanced_query': enhanced_query,
            'enhanced_support': enhanced_support,
            'intervention': intervention_query,
            'q_saliency': q_saliency,
            's_saliency': s_saliency,
            'contrast_loss': contrast_loss,
            'adversarial_loss': adversarial_loss,
            'reg_loss': reg_loss
        }