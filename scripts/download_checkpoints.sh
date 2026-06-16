#!/bin/bash
# Download pretrained CSSNet checkpoints
# Usage: bash scripts/download_checkpoints.sh

BASE_URL="https://github.com/chenshaobo/CSSNet/releases/download/v1.0"
CHECKPOINTS_DIR="checkpoints"

mkdir -p $CHECKPOINTS_DIR

echo "Downloading CSSNet checkpoints..."
echo "Please download manually from the GitHub Releases page or Google Drive"
echo "Checkpoints will be placed in: $CHECKPOINTS_DIR/"
echo ""
echo "Required checkpoints:"
echo "  - cssnet_rn50_pascal_f0_1shot.pt  (PASCAL-5^0, fold 0, 1-shot)"
echo "  - cssnet_rn50_pascal_f1_1shot.pt  (PASCAL-5^1, fold 1, 1-shot)"
echo "  - cssnet_rn50_pascal_f2_1shot.pt"
echo "  - cssnet_rn50_pascal_f3_1shot.pt"
echo "  - cssnet_rn50_coco_f0_1shot.pt   (COCO-20^0, fold 0, 1-shot)"
echo "  - cssnet_rn50_fss1000_1shot.pt   (FSS-1000, 1-shot)"
