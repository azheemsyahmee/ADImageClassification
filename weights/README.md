# Pretrained Model Checkpoint Weights

This directory stores PyTorch `.pth` checkpoint weights for the 4 evaluated CNN backbones.

## Model Checkpoint Inventory

| Model | Checkpoint File | Size | Storage Strategy |
| :--- | :--- | :---: | :--- |
| **GoogLeNet** | `googleNet.pth` | 22.5 MB | Committed in repository |
| **ResNet-18** | `resnet18.pth` | 44.7 MB | Committed in repository |
| **ResNet-50** | `resnet50.pth` | 94.3 MB | Available via [GitHub Releases](../../releases) / Git LFS |
| **VGG-16** | `vgg16.pth` | 537.1 MB | Available via [GitHub Releases](../../releases) / Git LFS |

## How to Download Additional Checkpoints

Due to file size constraints, `vgg16.pth` (537 MB) and `resnet50.pth` (94 MB) should be downloaded from the repository Releases page:
```bash
# Option A: Download from Releases and place into weights/
# Place resnet50.pth and vgg16.pth directly inside this directory.

# Option B: Use Git LFS if enabled on your remote
git lfs pull
```
