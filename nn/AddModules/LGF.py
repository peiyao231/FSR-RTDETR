import torch
import torch.nn as nn
import torch.nn.functional as F

# Ultralytics Conv (preferred). Fallback provided for standalone usage.
try:
    from ultralytics.nn.modules.conv import Conv
except Exception:
    class Conv(nn.Module):
        def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
            super().__init__()
            if p is None:
                p = k // 2
            self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))


class LGFGate(nn.Module):
    """
    Learnable gated combiner with a global scalar:
        alpha = sigmoid(γ),  y = alpha * p + (1-alpha) * i
    where p and i denote two aligned feature tensors.
    """
    def __init__(self, init_value: float = 0.0):
        super().__init__()
        # γ is a global learnable scalar; sigmoid(0)=0.5 gives neutral initialization
        self.γ = nn.Parameter(torch.tensor(float(init_value)).view(1, 1, 1, 1))

    def forward(self, p: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.γ)  # broadcast to all dims
        return alpha * p + (1.0 - alpha) * i


class LGF(nn.Module):
    """
    LGF: Learnable Gated Fusion across adjacent feature levels.

    Args:
        in_features: (c_low, c_high) channel dimensions of (x_low, x_high)
        out_features: output channels after fusion
        groups: number of channel groups for grouped fusion (default: 4)
    """
    def __init__(self, in_features, out_features, groups: int = 4) -> None:
        super().__init__()
        assert isinstance(in_features, (list, tuple)) and len(in_features) == 2, \
            "in_features must be a tuple/list like (c_low, c_high)"
        self.groups = int(groups)
        self.gate = LGFGate(init_value=0.0)

        c_low, c_high = int(in_features[0]), int(in_features[1])
        self.conv1x1 = Conv(c_low, c_high, k=1) if c_low != c_high else nn.Identity()
        self.tail_conv = Conv(c_high, out_features, k=3)

    def forward(self, x):
        # Expect x = (x_low, x_high)
        if not (isinstance(x, (list, tuple)) and len(x) == 2):
            raise TypeError("LGF forward expects input as (x_low, x_high).")

        x_low, x_high = x
        if x_low is None or x_high is None:
            raise ValueError("x_low and x_high must not be None.")

        # Align channels
        x_low = self.conv1x1(x_low)

        # Align spatial resolution to x_high
        tgt_size = x_high.shape[-2:]
        x_low = F.interpolate(x_low, size=tgt_size, mode="bilinear", align_corners=False)

        # Grouped fusion
        c = x_high.shape[1]
        assert c == x_low.shape[1], "Channel mismatch after conv1x1 alignment."
        assert c % self.groups == 0, f"Channels ({c}) must be divisible by groups ({self.groups})."

        xh = torch.chunk(x_high, self.groups, dim=1)
        xl = torch.chunk(x_low,  self.groups, dim=1)

        fused = [self.gate(xl[g], xh[g]) for g in range(self.groups)]
        y = torch.cat(fused, dim=1)

        # Output projection
        y = self.tail_conv(y)
        return y
