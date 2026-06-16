# CSSNet: Causal Saliency-Guided Structural Refinement for Few-Shot Semantic Segmentation

Official PyTorch implementation of **CSSNet**, accepted by **The Visual Computer**.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

CSSNet introduces a **Causal Saliency Foreground Module (CSFM)** and a **Structure Refinement Module (SRM)** into a correlation-based few-shot segmentation pipeline. The CSFM learns to suppress background interference via saliency-guided soft-mask intervention and adversarial perturbation, while the SRM adapts DenseCRF with gradient-driven pairwise potentials to refine segmentation boundaries during inference.

### Key Contributions

- **Causal Saliency Foreground Module (CSFM)** – learns spatial saliency maps from backbone features and applies a differentiable soft-mask intervention to sever the confounding path between background context and the segmentation prediction
- **Structure Refinement Module (SRM)** – adapts DenseCRF post-processing by replacing colour-based potentials with gradient-based ones, improving boundary continuity and spatial consistency
- **State-of-the-art performance** on FSS-1000 (87.3%/88.7% mIoU for 1-/5-shot) and competitive results on PASCAL-5ⁱ and COCO-20ⁱ

---

## Installation

### Option 1: Conda (recommended)

```bash
git clone https://github.com/chenshaobo/CSSNet.git
cd CSSNet
conda env create -f environment.yml
conda activate cssnet
```

### Option 2: Pip

```bash
git clone https://github.com/chenshaobo/CSSNet.git
cd CSSNet

conda create -n cssnet python=3.9
conda activate cssnet

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

All dataset files are expected under the path specified by `--datapath`. The split files provided under `data/splits/` are already included.

### PASCAL-5ⁱ
1. Download [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) devkit (train/val)
2. Extract to `data/VOCdevkit/VOC2012/`

```
data/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── ImageSets/
```

### COCO-20ⁱ
1. Download [COCO 2014](https://cocodataset.org/#download) train/val images and annotations
2. Extract to your data path

```
/path/to/coco2014/
├── train2014/
├── val2014/
└── annotations/
```

### FSS-1000
1. Download [FSS-1000](https://github.com/HKUSTCV/FSS-1000)
2. Extract to your data path

---

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

Training checkpoints are saved in `logs/` automatically.

### Testing

To evaluate a trained model with SRM post-processing (default):

```bash
python test.py \
    --backbone resnet50 \
    --benchmark pascal \
    --datapath /path/to/datasets \
    --fold 0 \
    --nshot 1 \
    --load /path/to/checkpoint.pt
```

To test **without** SRM (for SRM ablation):

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

---

## Pretrained Models

We provide pretrained CSSNet checkpoints for all benchmarks. Download the weights and place them under a `checkpoints/` directory.

| Model | Backbone | Benchmark | Fold | Setting | mIoU | FB-IoU | Download |
|-------|----------|-----------|------|---------|------|--------|----------|
| CSSNet | ResNet-50 | PASCAL-5⁰ | 0 | 1-shot | 69.69 | 83.63 | _coming soon_ |
| CSSNet | ResNet-50 | PASCAL-5¹ | 1 | 1-shot | 72.65 | 82.23 | _coming soon_ |
| CSSNet | ResNet-50 | PASCAL-5² | 2 | 1-shot | 63.12 | 71.14 | _coming soon_ |
| CSSNet | ResNet-50 | PASCAL-5³ | 3 | 1-shot | 64.98 | 77.22 | _coming soon_ |
| CSSNet | ResNet-50 | PASCAL-5⁰ | 0 | 5-shot | 74.00 | 86.21 | _coming soon_ |
| CSSNet | ResNet-50 | COCO-20⁰ | 0 | 1-shot | 39.02 | 67.12 | _coming soon_ |
| CSSNet | ResNet-50 | FSS-1000 | — | 1-shot | 87.27 | 91.95 | _coming soon_ |

> Checkpoints will be uploaded to the [GitHub Releases](https://github.com/chenshaobo/CSSNet/releases) page. Please check there or open an issue if you need them urgently.

---

## Results

### Main Results (ResNet-50)

| Benchmark | Setting | mIoU | FB-IoU |
|-----------|---------|------|--------|
| FSS-1000 | 1-shot | **87.27** | 91.95 |
| FSS-1000 | 5-shot | **88.69** | 93.03 |
| PASCAL-5ⁱ (4-fold mean) | 1-shot | **68.00** | 78.90 |
| PASCAL-5ⁱ (4-fold mean) | 5-shot | **71.80** | 82.90 |
| COCO-20ⁱ (4-fold mean) | 1-shot | **43.20** | 70.10 |
| COCO-20ⁱ (4-fold mean) | 5-shot | **49.33** | 72.30 |

### Ablation: Component-wise contribution (PASCAL-5⁰, 5-shot, ResNet-50)

| Configuration | mIoU |
|--------------|------|
| Baseline | 70.00 |
| + Saliency generation | 70.41 |
| + Soft-mask intervention | 70.81 |
| + Adversarial perturbation | 71.06 |
| + Sparsity regularisation | 71.23 |
| + Auxiliary heads (full CSFM) | 71.44 |
| + SRM (full CSSNet) | **71.80** |

---

## Project Structure

```
CSSNet/
├── train.py                  # Training script
├── test.py                   # Testing / evaluation script
├── requirements.txt          # Pip dependencies
├── environment.yml           # Conda environment
├── scripts/
│   └── download_checkpoints.sh
├── model/
│   ├── CSSNet.py             # Main model definition
│   ├── CSFM.py               # Causal Saliency Foreground Module
│   ├── learner.py            # Training utilities
│   └── base/
│       ├── CRM.py            # Correlation Reconstruction Module
│       ├── FEM.py            # Feature Enhancement Module
│       ├── srm.py            # Structure Refinement Module (DenseCRF)
│       ├── conv4d.py         # 4D convolution utilities
│       ├── correlation.py    # Correlation layer
│       └── feature.py        # Feature processing
├── common/
│   ├── evaluation.py         # Evaluation metrics
│   ├── logger.py             # Logging
│   ├── utils.py              # Helper functions
│   └── vis.py                # Visualization
└── data/
    ├── dataset.py            # Dataset base class
    ├── pascal.py             # PASCAL-5ⁱ dataset
    ├── coco.py               # COCO-20ⁱ dataset
    ├── fss.py                # FSS-1000 dataset
    └── splits/               # Train/val fold splits
```

---

## Citation

If you find this work useful for your research, please cite our paper published in **The Visual Computer**:

```bibtex
@article{chen2025cssnet,
  title={Causal Saliency-Guided Structural Refinement for Robust Few-Shot Semantic Segmentation},
  author={Chen, Shaobo and [full author list]},
  journal={The Visual Computer},
  volume={},
  number={},
  pages={},
  year={2025},
  publisher={Springer},
  doi={}
}
```

> Please replace `[full author list]` and volume/number/pages/doi with the final publication details once available.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
