# LMFLOSS: A Hybrid Loss for Imbalanced Medical Image Classification
**Please note that the code in this repository has not been verified and tested in detail, and is only suitable for binary classification image segmentation tasks.**

This is an unofficial PyTorch implementation of the Large Margin aware Focal (LMF) Loss as described in the paper: [**LMFLOSS: A Hybrid Loss For Imbalanced Medical Image Classification.**](https://arxiv.org/abs/2212.12741)

This repository provides a PyTorch implementation of the LMF Loss that combines Focal Loss and Label Distribution Aware Margin (LDAM) Loss, specifically designed for addressing class imbalance in medical image classification.

## Installation
You can use these loss functions by cloning this repository:

```bash
git https://github.com/brianhou0208/LMF_Loss_Pytorch.git
cd LMF_Loss_Pytorch
```
## Usage Example
Here is a simple example of how to use the LMF Loss in your PyTorch model:

```python
import torch
from loss import FocalLoss, LDAMLoss, LMFLoss

# Assume you have a model, input, and target
model = ...  # Your PyTorch model
inputs = ...  # Your input data, e.g., torch.randn(1, 1, 256, 256)
targets = ...  # Your target labels, e.g., torch.rand(1, 1, 256, 256)

# Initialize the loss functions
lmf_loss = LMFLoss()
ldam_loss = LDAMLoss()
focal_loss = FocalLoss(alpha=1.0, gamma=1.5)

print(f'LMFLoss :', lmf_loss(predictions, targets).item())
print(f'LDAMLoss :', ldam_loss(predictions, targets).item())
print(f'FocalLoss :', focal_loss(predictions, targets).item())

# Compute the loss
predictions = model(inputs)
loss = lmf_loss(predictions, targets)
loss.backward()  # Backpropagation

```
## Acknowledgements
Our code is adapted from:
* LDAM Loss : [official implementation of LDAM](https://github.com/kaidic/LDAM-DRW)
* Focal Loss : [pytorch/torchvision](https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py) and [fvcore](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py)
Thanks for these authors for their valuable works.

## Contributions
This is an open-source project, and contributions of any kind are welcome, including suggestions and bug reports.

## License
This project is licensed under the [MIT License.](https://github.com/brianhou0208/LMF_Loss_Pytorch/blob/main/LICENSE)
