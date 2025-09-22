# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    Simple CNN for MNIST:
      Input:  [B, 1, 28, 28]
      Output: [B, 10] (logits for digits 0..9)
    """
    def __init__(self, hidden_dim: int = 128, dropout_p: float = 0.2):
        super().__init__()

        # --- Feature extractor ---
        # Conv layer 1: 1 input channel (grayscale) -> 32 feature maps
        # kernel_size=3, padding=1 keeps H,W unchanged (28x28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # Conv layer 2: 32 -> 64 feature maps; still use k=3, p=1 to preserve size before pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Max pooling halves spatial size after each conv block: 28->14, then 14->7
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Classifier head ---
        # After two pools, feature map shape is [B, 64, 7, 7] => 64*7*7 = 3136 flattened features
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=10)  # 10 digits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: conv -> relu -> pool
        x = self.conv1(x)          # [B, 1, 28, 28] -> [B, 32, 28, 28]
        x = F.relu(x)
        x = self.pool(x)           # [B, 32, 28, 28] -> [B, 32, 14, 14]

        # Block 2: conv -> relu -> pool
        x = self.conv2(x)          # [B, 32, 14, 14] -> [B, 64, 14, 14]
        x = F.relu(x)
        x = self.pool(x)           # [B, 64, 14, 14] -> [B, 64, 7, 7]

        # Flatten keeping the batch dimension
        x = x.view(x.size(0), -1)  # [B, 64, 7, 7] -> [B, 3136]

        # Classifier
        x = self.fc1(x)            # [B, 3136] -> [B, hidden_dim]
        x = F.relu(x)
        x = self.dropout(x)        # regularization (active only in training mode)
        logits = self.fc2(x)       # [B, hidden_dim] -> [B, 10]
        return logits


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Quick sanity-check utility (optional) ---
if __name__ == "__main__":
    model = MNISTNet()
    print(model)
    print("Trainable params:", count_params(model))

    dummy = torch.randn(4, 1, 28, 28)  # batch of 4 fake MNIST images
    out = model(dummy)
    print("Output shape:", out.shape)   # should be [4, 10]
