# CSSNet: Causal Saliency-Guided Structural Refinement for Few-Shot Semantic Segmentation

Official PyTorch implementation of **CSSNet**, accepted by **The Visual Computer**.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

CSSNet introduces a **Causal Saliency Foreground Module (CSFM)** and a **Structure Refinement Module (SRM)** into a correlation-based few-shot segmentation pipeline. The CSFM learns to suppress background interference via saliency-guided soft-mask intervention and adversarial perturbation, while the SRM adapts DenseCRF with gradient-driven pairwise potentials to refine segmentation boundaries during inference.

### Key Contributions

- **Causal Saliency Foreground Module (CSFM)** – learns spatial saliency maps from backbone features and applies a differentiable soft-mask intervention to sever the confounding path between background context and the segmentation prediction
- **Structure Refinement Module (SRM)** – adapts DenseCRF post-processing by replacing colour-based potentials with gradient-based ones, improving boundary continuity and spatial consistency
- **State-of-the-art performance** on FSS-1000 (87.3%/88.7% mIoU for 1-/5-shot) and competitive results on PASCAL-5ⁱ and COCO-20ⁱ

## Installation

```bash
git clone https://github.com/chenshaobo/CSSNet.git
cd CSSNet

conda create -n cssnet python=3.9
conda activate cssnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Dataset Preparation

### PASCAL-5ⁱ
1. Download [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) devkit
2. Place it at `data/VOCdevkit/VOC2012/`
3. The split files are already provided under `data/splits/pascal/`

### COCO-20ⁱ
1. Download [COCO 2014](https://cocodataset.org/#download) train/val images and annotations
2. Place at your data path and update `--datapath` accordingly

### FSS-1000
1. Download [FSS-1000](https://github.com/HKUSTCV/FSS-1000)
2. Place at your data path and update `--datapath` accordingly

## Usage

### Training

To train CSSNet on PASCAL-5ⁱ fold 0 (1-shot, ResNet-50):

```bash
python train.py \
    --backbone resnet50 \
    --benchmark pascal \
    --datapath /path/to/datasets \
    --fold 0 \
    --nshot 1 \
    --bsz 8 \
    --lr 5e-4 \
    --max-epoch 50
```

### Testing

To evaluate a trained model:

```bash
python test.py \
    --backbone resnet50 \
    --benchmark pascal \
    --datapath /path/to/datasets \
    --fold 0 \
    --nshot 1 \
    --load /path/to/checkpoint.pt
```

To test **without** SRM post-processing (for SRM ablation):

```bash
python test.py \
    --backbone resnet50 \
    --benchmark pascal \
    --datapath /path/to/datasets \
    --fold 0 \
    --nshot 1 \
    --load /path/to/checkpoint.pt \
    --no-srm
```

## Pretrained Models

| Model | Backbone | Benchmark | Fold | Setting | mIoU | FB-IoU | Download |
|-------|----------|-----------|------|---------|------|--------|----------|
| CSSNet | ResNet-50 | PASCAL-5⁰ | 0 | 1-shot | 69.69 | 83.63 | [Link]() |
| CSSNet | ResNet-50 | PASCAL-5⁰ | 0 | 5-shot | 74.00 | 86.21 | [Link]() |
| CSSNet | ResNet-50 | COCO-20⁰ | 0 | 1-shot | 39.02 | 67.12 | [Link]() |
| CSSNet | ResNet-50 | FSS-1000 | - | 1-shot | 87.27 | 91.95 | [Link]() |

> Checkpoints will be uploaded soon. Please check back or open an issue if you need them urgently.

## Results

### Main Results (ResNet-50, 1-shot)

| Benchmark | mIoU |
|-----------|------|
| FSS-1000 | **87.27** |
| PASCAL-5ⁱ (mean) | 68.00 |
| COCO-20ⁱ (mean) | 43.20 |

## Ablation

### Component-wise contribution (PASCAL-5⁰, 5-shot, ResNet-50)

| Configuration | mIoU |
|--------------|------|
| Baseline | 70.00 |
| + Saliency generation | 70.41 |
| + Soft-mask intervention | 70.81 |
| + Adversarial perturbation | 71.06 |
| + Sparsity regularisation | 71.23 |
| + Auxiliary heads (full CSFM) | 71.44 |
| + SRM (full CSSNet) | **71.80** |

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2025cssnet,
  title={Causal Saliency-Guided Structural Refinement for Robust Few-Shot Semantic Segmentation},
  author={Chen, Shaobo and ...},
  journal={The Visual Computer},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
