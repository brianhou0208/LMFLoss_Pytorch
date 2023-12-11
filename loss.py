import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss function.
    Focal Loss for Dense Object Detection
    paper : https://arxiv.org/abs/1708.02002
    code from : https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py

    Args:
        alpha (float, optional): The weight factor for the positive class. Defaults to 0.25.
        gamma (float, optional): The focusing parameter. Defaults to 2.
        reduction (str, optional): Specifies the reduction method for the loss. Defaults to "mean".

    Returns:
        torch.Tensor: The Focal loss.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Focal loss.

        Args:
            inputs (torch.Tensor): A tensor of shape (B, C, H, W) representing the model's predictions.
            targets (torch.Tensor): A tensor of shape (B, C, H, W) representing the ground truth labels.

        Returns:
            torch.Tensor: The Focal loss.
        """
        # Calculate the binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        # Calculate the probability based on the sigmoid function
        probs = torch.sigmoid(inputs)
        # Calculate the focal loss components
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        # Reduce the loss if specified.
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()

        return focal_loss

class LDAMLoss(nn.Module):
    """LDAM loss function.
    Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
    paper : https://arxiv.org/abs/1906.07413
    code from : https://github.com/kaidic/LDAM-DRW/blob/master/losses.py

    Args:
        max_m (float, optional): The maximum margin. Defaults to 0.5.
        s (float, optional): The scaling factor. Defaults to 30.

    Returns:
        torch.Tensor: The LDAM loss.
    """

    def __init__(self, max_m: float = 0.5, s: float = 30):
        super().__init__()
        self.cls_num_list = None
        self.m_list = None
        self.max_m = max_m
        self.s = s

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the LDAM loss.

        Args:
            inputs (torch.Tensor): A tensor of shape (B, C, H, W) representing the model's predictions.
            targets (torch.Tensor): A tensor of shape (B, C, H, W) representing the ground truth labels.

        Returns:
            torch.Tensor: The LDAM loss.
        """
        cls_num_list = self.calculate_batch_class_distribution(targets)
        m_list = self.calculate_class_margins(cls_num_list, self.max_m)


        margin = m_list[1] * targets.float()
        margin_adjusted_inputs = inputs - margin

        output = torch.where(targets == 1, margin_adjusted_inputs, inputs)

        return F.binary_cross_entropy_with_logits(self.s * output, targets.float())

    @staticmethod
    def calculate_batch_class_distribution(targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the number of pixels for each class within a batch.

        Args:
        - targets (torch.Tensor): A tensor of shape (B, C, H, W)
          containing binary segmentation masks.

        Returns:
        - cls_num_list (torch.Tensor): A tensor with the number of pixels for each class within the batch.
        """
        # Assuming binary classification (0 for background, 1 for foreground)
        background_count = (targets == 0).sum().item()
        foreground_count = (targets == 1).sum().item()

        # Create the class number list tensor for the batch
        cls_num_list = torch.Tensor([background_count, foreground_count])
        
        return cls_num_list

    @staticmethod
    def calculate_class_margins(cls_num_list: torch.Tensor, max_m: float) -> torch.Tensor:
        """Calculates the class-dependent margins.

        Args:
            cls_num_list (torch.Tensor): A tensor containing the number of pixels for each class within the batch.
            max_m (float): The maximum margin.

        Returns:
            torch.Tensor: A tensor containing the class-dependent margins.
        """

        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))

        return m_list

class LMFLoss(nn.Module):
    """LMF loss function.
    LMFLOSS: A Hybrid Loss For Imbalanced Medical Image Classification
    paper : https://arxiv.org/abs/2212.12741
    
    Args:
        max_m (float, optional): The maximum margin. Defaults to 0.5.
        s (float, optional): The scaling factor. Defaults to 30.
        alpha (float, optional): The weight factor for the positive class. Defaults to 1.
        gamma (float, optional): The focusing parameter. Defaults to 1.5.
        reduction (str, optional): Specifies the reduction method for the loss. Defaults to "mean".

    Returns:
        torch.Tensor: The LMF loss.
    """

    def __init__(self, max_m: float = 0.5, s: float = 30, alpha: float = 1.0, gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()

        self.ldam = LDAMLoss(max_m, s)
        self.focal = FocalLoss(alpha, gamma, reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the LMF loss.

        Args:
            inputs (torch.Tensor): A tensor of shape (B, C, H, W) representing the model's predictions.
            targets (torch.Tensor): A tensor of shape (B, C, H, W) representing the ground truth labels.

        Returns:
            torch.Tensor: The LMF loss.
        """
        loss = self.ldam(inputs, targets) + self.focal(inputs, targets)

        return loss
    


if __name__ == '__main__':
    
    # Example usage:
    B, C, H, W = 5, 1, 512, 512
    batch_image = torch.randn(B, C, H, W)
    # Let's say we have a batch of masks tensors
    batch_masks = torch.rand(B, C, H, W) > 0.5  # Generate dummy masks for demonstration
    batch_masks = batch_masks.long()

    batch_image, batch_masks = batch_image.float(), batch_masks.float()

    loss = LDAMLoss()
    out = loss(batch_image, batch_masks)
    print(f'LDAMLoss :', out.item())

    loss = FocalLoss(alpha=1, gamma=1.5)
    out = loss(batch_image, batch_masks)
    print(f'FocalLoss :', out.item())

    loss = LMFLoss()
    out = loss(batch_image, batch_masks)
    print(f'LMFLoss :', out.item())
    