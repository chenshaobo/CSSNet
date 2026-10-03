# Attention-control experiment: equal-parameter SE branch

Control experiment for the revision of *Causal Saliency-Guided Structural Refinement
for Robust Few-Shot Semantic Segmentation*.

## Purpose

The CSFM spatial saliency branch is replaced by an equal-parameter SE channel-attention
block, in order to test whether the gain of CSSNet can be reproduced by conventional
channel attention. If the improvement came from generic attention enhancement rather
than from the proposed spatial saliency intervention, the SE variant should match CSSNet.

## Configuration

| Item | Value |
| --- | --- |
| Dataset | PASCAL-5^0, fold 0 |
| Setting | 1-shot |
| Backbone | ResNet-50 (ImageNet pre-trained) |
| Branch switch | ABL_SALIENCY=se (SEBranch in model/CSFM.py) |
| Batch size | 6 |
| Learning rate | 5e-4 |
| Epochs | 50 |
| Weight decay | 0.05 |
| Mixed precision | on |
| Hardware | single GPU |

Parameter count at the first level: 16,384 (SE branch) versus 16,449 (learned saliency
branch), a difference of 0.4%.

The two branches differ in the shape of their output. The SE branch produces a
channel-wise gate of shape [B, C, 1, 1], which re-weights entire feature channels
uniformly across all spatial positions. The learned saliency branch produces a spatial
map of shape [B, 1, H, W], which assigns an independent weight to every location.

## Commands

Training:

    ABL_SALIENCY=se python -X utf8 train.py --datapath data/VOCdevkit --benchmark pascal --backbone resnet50 --fold 0 --bsz 6 --niter 50 --logpath E5_SE

Evaluation with SRM (the default of test.py):

    python -X utf8 test.py --datapath data/VOCdevkit --benchmark pascal --backbone resnet50 --fold 0 --nshot 1 --bsz 8 --load logs/E5_SE.log/best_model.pt

## Results

PASCAL-5^0, fold 0, 1-shot, ResNet-50.

| Setting | mIoU | FB-IoU |
| --- | --- | --- |
| SE attention control, no SRM (best validation epoch, 41 of 50) | 67.58 | 83.11 |
| SE attention control, with SRM | 67.68 | 83.50 |

For reference, obtained under the same fold and setting:

| Setting | mIoU | FB-IoU |
| --- | --- | --- |
| CSSNet, no CRF | 68.93 | 82.95 |
| CSSNet, with SRM (gradient bilateral kernel) | 69.69 | 83.63 |

The equal-parameter SE control reaches 67.68 mIoU, below the FECANet baseline and well
below CSSNet. Adding SRM changes it by +0.10 pp, against +0.76 pp for CSSNet, so the
boundary-refinement module does not compensate for the loss of spatial selectivity.

## Files

- results/E5_SE_train_log.txt -- full training log, 50 epochs.
- results/E5_SE_test_with_SRM_log.txt -- evaluation log with SRM.
- results/summary.json -- the figures above in machine-readable form.

## Notes

In this codebase the validation split and the test split consist of the same episodes
(data/pascal.py), which is the established protocol of the repository; the best
validation figure is therefore directly comparable with the test figures above.

The trained checkpoint is not stored in this repository.
