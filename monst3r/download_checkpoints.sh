#!/bin/bash
# Download all required checkpoints for MonST3R data preparation
# These are third-party model weights that are publicly available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Downloading MonST3R checkpoint ==="
# MonST3R ViT-Large model (2.2GB)
# Source: https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt
mkdir -p checkpoints
if [ ! -f checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth ]; then
    gdown --fuzzy https://drive.google.com/file/d/1Z1jO_JmfZj0z3bgMvCwqfUhyZ1bIbc9E/view?usp=sharing -O checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth
    echo "Done."
else
    echo "Already exists, skipping."
fi

echo "=== Downloading SAM2 checkpoint ==="
# SAM2.1 Hiera-Large (857MB)
# Source: https://github.com/facebookresearch/sam2
mkdir -p third_party/sam2/checkpoints
if [ ! -f third_party/sam2/checkpoints/sam2.1_hiera_large.pt ]; then
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt
    echo "Done."
else
    echo "Already exists, skipping."
fi

echo "=== Downloading RAFT (Sea-RAFT) checkpoint ==="
# Tartan-C-T-TSKH-spring540x960-M (76MB)
mkdir -p third_party/RAFT/models
if [ ! -f third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth ]; then
    gdown --fuzzy https://drive.google.com/file/d/1a0C5FTdhjM4rKrfXiGhec7eq2YM141lu/view?usp=drive_link -O third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth
    echo "Done."
else
    echo "Already exists, skipping."
fi

echo "=== Note ==="
echo "ResNet34 weights (resnet34-b627a593.pth) will be automatically"
echo "downloaded by torchvision at runtime. No manual download needed."
echo ""
echo "All checkpoints downloaded successfully!"
