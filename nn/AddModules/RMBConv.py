import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Conv import (Ultralytics preferred) + fallback
# ----------------------------
try:
    # Ultralytics style
    from ultralytics.nn.modules.conv import Conv
except Exception:
    try:
        # your original relative import (if placed under ultralytics/nn/...)
        from ..modules.conv import Conv  # type: ignore
    except Exception:
        # minimal fallback for standalone debug
        class Conv(nn.Module):
            def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=False):
                super().__init__()
                self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
                self.bn = nn.BatchNorm2d(c2)
                self.act = nn.Identity() if not act else nn.SiLU(inplace=True)

            def forward(self, x):
                return self.act(self.bn(self.conv(x)))


# ----------------------------
# Helper: fuse Conv+BN to (kernel, bias)
# ----------------------------
def _fuse_ultralytics_conv_bn(m: Conv):
    """
    Fuse Ultralytics Conv (Conv2d + BN, act disabled) into a single conv kernel & bias.
    """
    w = m.conv.weight
    # Ultralytics conv often uses bias=False -> treat as zeros
    b = m.conv.bias if (m.conv.bias is not None) else torch.zeros(w.size(0), device=w.device, dtype=w.dtype)

    rm = m.bn.running_mean
    rv = m.bn.running_var
    gamma = m.bn.weight
    beta = m.bn.bias
    eps = m.bn.eps

    std = torch.sqrt(rv + eps)
    t = (gamma / std).reshape(-1, 1, 1, 1)
    w_fused = w * t
    b_fused = beta + (b - rm) * gamma / std
    return w_fused, b_fused


def _pad_1x1_to_3x3(w_1x1: torch.Tensor):
    if w_1x1 is None:
        return 0
    return F.pad(w_1x1, [1, 1, 1, 1])


# ----------------------------
# Re-parameterizable branches
# ----------------------------
class RMBBranch1x1(nn.Module):
    """
    1x1 -> 1x1 (both Conv+BN, act disabled), can be fused to a single 1x1 conv.
    """

    def __init__(self, in_channels: int, out_channels: int, stride=1, padding=0, deploy: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = stride
        self.padding = padding
        self.deploy = bool(deploy)

        if self.deploy:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1,
                                  stride=self.stride, padding=self.padding, bias=True)
        else:
            self.conv1 = Conv(self.in_channels, self.out_channels, k=1, s=self.stride, p=self.padding, act=False)
            self.conv2 = Conv(self.out_channels, self.out_channels, k=1, s=1, p=0, act=False)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def switch_to_deploy(self):
        if self.deploy:
            return
        k1, b1 = _fuse_ultralytics_conv_bn(self.conv1)
        k2, b2 = _fuse_ultralytics_conv_bn(self.conv2)

        # fuse 1x1 then 1x1: W = W2 @ W1
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1,
                              stride=self.stride, padding=self.padding, bias=True)
        # k2: [O, I, 1, 1], k1: [I, Cin, 1, 1] -> [O, Cin, 1, 1]
        self.conv.weight.data = torch.einsum('oi,icjk->ocjk', k2.squeeze(-1).squeeze(-1), k1)
        self.conv.bias.data = b2 + (b1.view(1, -1, 1, 1) * k2).sum(dim=(1, 2, 3))

        del self.conv1, self.conv2
        self.deploy = True


class RMBBranch3x3(nn.Module):
    """
    3x3 -> 1x1 (both Conv+BN, act disabled), can be fused to a single 3x3 conv.
    """

    def __init__(self, in_channels: int, out_channels: int, stride=1, padding=1, deploy: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = stride
        self.padding = padding
        self.deploy = bool(deploy)

        if self.deploy:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3,
                                  stride=self.stride, padding=self.padding, bias=True)
        else:
            self.conv1 = Conv(self.in_channels, self.out_channels, k=3, s=self.stride, p=self.padding, act=False)
            self.conv2 = Conv(self.out_channels, self.out_channels, k=1, s=1, p=0, act=False)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def switch_to_deploy(self):
        if self.deploy:
            return
        k1, b1 = _fuse_ultralytics_conv_bn(self.conv1)  # 3x3
        k2, b2 = _fuse_ultralytics_conv_bn(self.conv2)  # 1x1

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3,
                              stride=self.stride, padding=self.padding, bias=True)
        # k2: [O, I, 1, 1] applied on output of k1 => equivalent 3x3
        self.conv.weight.data = torch.einsum('oi,icjk->ocjk', k2.squeeze(-1).squeeze(-1), k1)
        self.conv.bias.data = b2 + (b1.view(1, -1, 1, 1) * k2).sum(dim=(1, 2, 3))

        del self.conv1, self.conv2
        self.deploy = True


# ----------------------------
# RMBConv (your GCConv renamed & cleaned)
# ----------------------------
class RMBConv(nn.Module):
    """
    RMBConv: Re-parameterized Multi-Branch Convolution.

    Train-time:
        y = SiLU( B3(x) + B3'(x) + B1(x) + BN_id(x) )
        - two 3x3->1x1 branches
        - one 1x1->1x1 branch
        - optional identity BN (only if Cin=Cout and stride=1)
    Deploy-time:
        fold into a single 3x3 Conv2d with bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        padding_mode: str = "zeros",
        deploy: bool = False
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.deploy = bool(deploy)

        assert self.kernel_size == 3 and self.padding == 1, "RMBConv assumes 3x3, padding=1."
        padding_11 = self.padding - self.kernel_size // 2  # -> 0

        self.act = nn.SiLU(inplace=True)

        if self.deploy:
            self.reparam_3x3 = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=3,
                stride=self.stride, padding=self.padding, bias=True,
                padding_mode=padding_mode
            )
        else:
            self.path_residual = nn.BatchNorm2d(self.in_channels) if (self.out_channels == self.in_channels and self.stride == 1) else None

            self.path_3x3_1 = RMBBranch3x3(self.in_channels, self.out_channels, stride=self.stride, padding=self.padding, deploy=False)
            self.path_3x3_2 = RMBBranch3x3(self.in_channels, self.out_channels, stride=self.stride, padding=self.padding, deploy=False)
            self.path_1x1   = RMBBranch1x1(self.in_channels, self.out_channels, stride=self.stride, padding=padding_11, deploy=False)

            self.id_tensor = None  # built lazily when fusing identity BN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_3x3"):
            return self.act(self.reparam_3x3(x))

        id_out = 0 if (self.path_residual is None) else self.path_residual(x)
        y = self.path_3x3_1(x) + self.path_3x3_2(x) + self.path_1x1(x) + id_out
        return self.act(y)

    def _fuse_bn_identity(self, bn: nn.Module):
        """
        Fuse identity BN into (kernel, bias) with 3x3 identity kernel.
        """
        if bn is None:
            return 0, 0
        assert isinstance(bn, (nn.BatchNorm2d, nn.SyncBatchNorm))

        if self.id_tensor is None:
            k = np.zeros((self.in_channels, self.in_channels, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                k[i, i, 1, 1] = 1.0
            self.id_tensor = torch.from_numpy(k).to(bn.weight.device)

        kernel = self.id_tensor
        rm = bn.running_mean
        rv = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(rv + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        kernel_fused = kernel * t
        bias_fused = beta - rm * gamma / std
        return kernel_fused, bias_fused

    def get_equivalent_kernel_bias(self):
        """
        Compute equivalent (3x3 kernel, bias) for deploy.
        """
        # fuse branch modules to deploy to extract their conv params
        self.path_3x3_1.switch_to_deploy()
        k31, b31 = self.path_3x3_1.conv.weight.data, self.path_3x3_1.conv.bias.data

        self.path_3x3_2.switch_to_deploy()
        k32, b32 = self.path_3x3_2.conv.weight.data, self.path_3x3_2.conv.bias.data

        self.path_1x1.switch_to_deploy()
        k11, b11 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data

        kid, bid = self._fuse_bn_identity(self.path_residual)

        kernel = k31 + k32 + _pad_1x1_to_3x3(k11) + kid
        bias = b31 + b32 + b11 + bid
        return kernel, bias

    def switch_to_deploy(self):
        """
        Convert to deploy mode: fold all branches into reparam_3x3.
        """
        if hasattr(self, "reparam_3x3"):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_3x3 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3,
            stride=self.stride, padding=self.padding, bias=True
        )
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias

        # delete training-time branches
        del self.path_3x3_1, self.path_3x3_2, self.path_1x1
        if hasattr(self, "path_residual"):
            del self.path_residual
        if hasattr(self, "id_tensor"):
            del self.id_tensor

        self.deploy = True
