import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=512, freeze=True):
        super().__init__()
        # Load pretrained ResNet18 and remove the final classifier
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # remove FC
        self.output_dim = output_dim

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: (B, K, C, H, W)
        return: (B, K, output_dim)
        """
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)                 # (B*K, C, H, W)
        feats = self.backbone(x)                   # (B*K, 512, 1, 1)
        feats = feats.view(B, K, -1)               # (B, K, 512)
        return feats
    


