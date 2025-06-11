# dino_encoder.py  ──────────────────────────────────────────────────────────────
import timm
import torch.nn as nn

class DINOEncoder(nn.Module):
    """
    DINO ViT-S/16 backbone (384-D CLS token).
    The model is frozen by default so it behaves like a fixed feature extractor.
    """
    def __init__(self,
                 model_name: str = "vit_small_patch16_224",
                 pretrained_tag: str = "dino",   # timm will pull the DINO weights
                 freeze: bool = True):
        super().__init__()

        # ---- create_model -----------------------------------------------------
        # • num_classes=0 avoids the classifier head
        # • global_pool='token' returns the CLS token (384-D for ViT-S)
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            pretrained_cfg={"tag": pretrained_tag},
            num_classes=0,
            global_pool="token",
        )
        self.embed_dim = self.backbone.num_features        # 384

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()                           # lock BN & dropout

    def forward(self, x):
        """
        x : (B, 3, H, W) in [0, 1] – will be resized/centre-cropped to 224×224
        Returns (B, 384) CLS token features.
        """
        return self.backbone(x)
