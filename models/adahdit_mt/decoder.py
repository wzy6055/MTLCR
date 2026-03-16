import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1, act=True, use_syncbn=False):
        super().__init__()
        bn = nn.SyncBatchNorm if use_syncbn else nn.BatchNorm2d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = bn(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    """ 2× 上采样 + 3×3卷积细化（避免棋盘格） """
    def __init__(self, ch, use_syncbn=False, up_mode="bilinear"):
        super().__init__()
        self.up_mode = up_mode
        self.refine = ConvBNAct(ch, ch, k=3, p=1, use_syncbn=use_syncbn)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode=self.up_mode, align_corners=False)
        x = self.refine(x)
        return x

class ProgressiveUpDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        mid_channels: int = 256,
        use_syncbn: bool = False,
        up_mode: str = "bilinear",
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.up_mode = up_mode
        self.stem = ConvBNAct(in_channels, mid_channels, k=1, p=0, use_syncbn=use_syncbn)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()

        self.up_blocks = nn.ModuleList([UpBlock(mid_channels, use_syncbn=use_syncbn, up_mode=up_mode) for _ in range(4)])

        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Expect (B, C, Hs, Ws)"

        x = self.stem(x)        # (B, mid, Hs, Ws)
        x = self.dropout(x)

        for block in self.up_blocks:
            x = block(x)

        logits = self.classifier(x)  # (B, num_classes, H, W)
        return logits


if __name__ == "__main__":
    x = torch.randn(8, 1024, 16, 16)

    decoder = ProgressiveUpDecoder(in_channels=1024, num_classes=10, mid_channels=256)
    y = decoder(x)
    print(y.shape)