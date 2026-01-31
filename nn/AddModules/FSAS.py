# -*- coding: utf-8 -*-
"""
FSAS (Frequency-Selective Adaptive Sampling) = FFS + RADDC
- FFS: Frequency-aware Feature Selection
- RADDC: Radial Adaptive Dilated Deformable Convolution

Cleaned & renamed version for Ultralytics/RT-DETR integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# MMCV Deformable Conv (Required for RADDC)
# ----------------------------
try:
    from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d, ModulatedDeformConv2d
except Exception as e:
    modulated_deform_conv2d = None
    ModulatedDeformConv2d = None


# ----------------------------
# Helper: Laplacian pyramid
# ----------------------------
def generate_laplacian_pyramid(x: torch.Tensor, num_levels: int, size_align: bool = True,
                              mode: str = "bilinear") -> list[torch.Tensor]:
    """
    Build Laplacian pyramid (aligned to input H,W if size_align=True).
    """
    pyr = []
    cur = x
    _, _, H, W = cur.shape

    for _ in range(num_levels):
        b, c, h, w = cur.shape
        down = F.interpolate(cur, (h // 2 + h % 2, w // 2 + w % 2),
                             mode=mode, align_corners=(H % 2) == 1)
        if size_align:
            up = F.interpolate(down, (H, W), mode=mode, align_corners=(H % 2) == 1)
            lap = F.interpolate(cur, (H, W), mode=mode, align_corners=(H % 2) == 1) - up
        else:
            up = F.interpolate(down, (h, w), mode=mode, align_corners=(H % 2) == 1)
            lap = cur - up

        pyr.append(lap)
        cur = down

    if size_align:
        cur = F.interpolate(cur, (H, W), mode=mode, align_corners=(H % 2) == 1)
    pyr.append(cur)
    return pyr


# ----------------------------
# FFS: Frequency-aware Feature Selection
# ----------------------------
class FFS(nn.Module):
    """
    Frequency-aware Feature Selection (FFS).
    It decomposes features into frequency bands and applies learnable spatial weights per band.
    """

    def __init__(
        self,
        in_channels: int,
        k_list: list[int] = (2,),
        lowfreq_att: bool = True,
        lp_type: str = "freq",          # {"freq", "avgpool", "laplacian"}
        act: str = "sigmoid",           # {"sigmoid", "softmax"}
        spatial_kernel: int = 3,
        spatial_group: int = 1,
        init: str = "zero",
        global_selection: bool = False,  # only used in lp_type="freq"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.k_list = list(k_list)
        self.lowfreq_att = lowfreq_att
        self.lp_type = lp_type
        self.act = act
        self.spatial_group = in_channels if spatial_group > 64 else spatial_group
        self.global_selection = global_selection

        # band-wise spatial weight convs
        n_band = len(self.k_list) + (1 if self.lowfreq_att else 0)
        self.weight_convs = nn.ModuleList()
        for _ in range(n_band):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.spatial_group,
                kernel_size=spatial_kernel,
                stride=1,
                padding=spatial_kernel // 2,
                groups=self.spatial_group,
                bias=True,
            )
            if init == "zero":
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
            self.weight_convs.append(conv)

        # avgpool mode needs explicit LP filters
        self.avg_lp = nn.ModuleList()
        if self.lp_type == "avgpool":
            for k in self.k_list:
                self.avg_lp.append(
                    nn.Sequential(
                        nn.ReplicationPad2d(k // 2),
                        nn.AvgPool2d(kernel_size=k, stride=1, padding=0),
                    )
                )
        elif self.lp_type in ("laplacian", "freq"):
            pass
        else:
            raise ValueError(f"Unsupported lp_type: {self.lp_type}")

        # optional global selection for FFT real/imag
        if self.global_selection and self.lp_type == "freq":
            self.global_real = nn.Conv2d(
                in_channels=in_channels, out_channels=self.spatial_group,
                kernel_size=1, stride=1, padding=0, groups=self.spatial_group, bias=True
            )
            self.global_imag = nn.Conv2d(
                in_channels=in_channels, out_channels=self.spatial_group,
                kernel_size=1, stride=1, padding=0, groups=self.spatial_group, bias=True
            )
            if init == "zero":
                nn.init.zeros_(self.global_real.weight); nn.init.zeros_(self.global_real.bias)
                nn.init.zeros_(self.global_imag.weight); nn.init.zeros_(self.global_imag.bias)

    def _sp_act(self, w: torch.Tensor) -> torch.Tensor:
        if self.act == "sigmoid":
            return w.sigmoid() * 2.0
        if self.act == "softmax":
            return w.softmax(dim=1) * w.shape[1]
        raise ValueError(f"Unsupported act: {self.act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if att_feat is None:
            att_feat = x
        b, c, h, w = x.shape
        x_list = []

        if self.lp_type == "avgpool":
            pre = x
            for idx, lp in enumerate(self.avg_lp):
                low = lp(pre)
                high = pre - low
                pre = low
                # w_band = self._sp_act(self.weight_convs[idx](x))
                w_band = self._sp_act(self.weight_convs[idx](att_feat))

                tmp = w_band.reshape(b, self.spatial_group, -1, h, w) * high.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))

            if self.lowfreq_att:
                w_low = self._sp_act(self.weight_convs[len(x_list)](x))
                tmp = w_low.reshape(b, self.spatial_group, -1, h, w) * pre.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pre)

        elif self.lp_type == "laplacian":
            pyr = generate_laplacian_pyramid(x, len(self.k_list), size_align=True)
            for idx in range(len(self.k_list)):
                high = pyr[idx]
                # w_band = self._sp_act(self.weight_convs[idx](x))
                w_band = self._sp_act(self.weight_convs[idx](att_feat))
                tmp = w_band.reshape(b, self.spatial_group, -1, h, w) * high.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))

            if self.lowfreq_att:
                w_low = self._sp_act(self.weight_convs[len(x_list)](x))
                low = pyr[-1]
                tmp = w_low.reshape(b, self.spatial_group, -1, h, w) * low.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pyr[-1])

        elif self.lp_type == "freq":
            pre = x
            x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))

            if self.global_selection:
                xr, xi = x_fft.real, x_fft.imag
                gr = self._sp_act(self.global_real(xr)).reshape(b, self.spatial_group, -1, h, w)
                gi = self._sp_act(self.global_imag(xi)).reshape(b, self.spatial_group, -1, h, w)
                xr = xr.reshape(b, self.spatial_group, -1, h, w) * gr
                xi = xi.reshape(b, self.spatial_group, -1, h, w) * gi
                x_fft = torch.complex(xr, xi).reshape(b, -1, h, w)

            for idx, freq in enumerate(self.k_list):
                # low-pass mask (center square)
                mask = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
                hs = round(h / 2 - h / (2 * freq))
                he = round(h / 2 + h / (2 * freq))
                ws = round(w / 2 - w / (2 * freq))
                we = round(w / 2 + w / (2 * freq))
                mask[:, :, hs:he, ws:we] = 1.0

                low = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask), norm="ortho").real
                high = pre - low
                pre = low

                # w_band = self._sp_act(self.weight_convs[idx](x))
                w_band = self._sp_act(self.weight_convs[idx](att_feat))
                tmp = w_band.reshape(b, self.spatial_group, -1, h, w) * high.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))

            if self.lowfreq_att:
                w_low = self._sp_act(self.weight_convs[len(x_list)](x))
                tmp = w_low.reshape(b, self.spatial_group, -1, h, w) * pre.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pre)

        return sum(x_list)


# ----------------------------
# RADDC: Radial Adaptive Dilated Deformable Convolution
# ----------------------------
class RADDC(nn.Module):
    """
    RADDC implements grid-aligned canonical offsets multiplied by a learned radial scaling map.
    It predicts:
      - scale s(x): (B, deform_groups, H, W)  -> multiplies canonical offsets
      - mask m(x):  (B, deform_groups*k*k, H, W)
    and applies modulated deform conv with dilation=1 (effective dilation comes from scale).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        base_dilation: int = 1,
        epsilon: float = 0.0,
        use_zero_dilation: bool = False,
    ):
        super().__init__()
        if ModulatedDeformConv2d is None or modulated_deform_conv2d is None:
            raise ImportError(
                "mmcv is required for RADDC (ModulatedDeformConv2d). "
                "Please install mmcv/mmcv-full matching your CUDA/PyTorch."
            )

        assert kernel_size == 3, "This cleaned RADDC version assumes 3x3 canonical lattice. Extend if you need 7x7."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.groups = groups
        self.deform_groups = deform_groups
        self.base_dilation = float(base_dilation)
        self.epsilon = float(epsilon)
        self.use_zero_dilation = bool(use_zero_dilation)

        # Deformable conv params (weight/bias)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        # radial scaling predictor: output (B, deform_groups, H, W)
        self.conv_scale = nn.Conv2d(
            in_channels=in_channels,
            out_channels=deform_groups,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        nn.init.zeros_(self.conv_scale.weight)
        # initialize scale bias near (d-1)/d to mimic your original behavior
        self.conv_scale.bias.data.fill_((base_dilation - 1.0) / max(base_dilation, 1.0) + self.epsilon)

        # modulation mask predictor: output (B, deform_groups*k*k, H, W)
        self.conv_mask = nn.Conv2d(
            in_channels=in_channels,
            out_channels=deform_groups * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        nn.init.zeros_(self.conv_mask.weight)
        nn.init.zeros_(self.conv_mask.bias)

        # canonical 3x3 lattice offsets v_k (y,x) flattened as [y0,x0, ..., y8,x8]
        offset = torch.tensor(
            [-1, -1,  -1, 0,  -1, 1,
              0, -1,   0, 0,   0, 1,
              1, -1,   1, 0,   1, 1],
            dtype=torch.float32
        )
        # shape: (1, deform_groups, 18, 1, 1)
        self.register_buffer("canonical_offset", offset.view(1, 1, -1, 1, 1).repeat(1, deform_groups, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape

        # scale: (B, G, H, W)
        scale = self.conv_scale(x)
        if self.use_zero_dilation:
            # ensure >= 0 and allow exact zero
            scale = (F.relu(scale + 1.0, inplace=True) - 1.0) * self.base_dilation
        else:
            # ensure > 0
            scale = F.relu(scale, inplace=True) * self.base_dilation

        # build full offsets: (B, G, 18, H, W) -> (B, 18*G, H, W)
        _, g, h, w = scale.shape
        scale = scale.view(b, g, 1, h, w)
        offset = (scale * self.canonical_offset).view(b, g * 18, h, w)

        # mask: (B, G*k*k, H, W)
        mask = torch.sigmoid(self.conv_mask(x))

        # apply modulated deform conv
        y = modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias,
            self.stride, self.padding,
            (1, 1),                 # dilation fixed to 1; effective dilation is absorbed by offsets
            self.groups,
            self.deform_groups
        )
        return y


# ----------------------------
# FSAS = FFS + RADDC
# ----------------------------
class FSAS(nn.Module):
    """
    FSAS module used to replace the 3x3 conv in backbone blocks.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        base_dilation: int = 1,
        ffs_cfg: dict | None = None,
        use_ffs: bool = True,
    ):
        super().__init__()
        self.use_ffs = use_ffs
        self.ffs = FFS(channels, **(ffs_cfg or {})) if use_ffs else nn.Identity()
        self.raddc = RADDC(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            deform_groups=deform_groups,
            base_dilation=base_dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffs(x)
        x = self.raddc(x)
        return x


# ----------------------------
# Optional: integrate into ResNet BasicBlock
# ----------------------------
# NOTE: Adjust BasicBlock import to your actual ultralytics version if needed.
try:
    from ultralytics.nn.modules.block import BasicBlock
except Exception:
    BasicBlock = None


class BasicBlock_FSAS(BasicBlock if BasicBlock is not None else nn.Module):
    """
    Replace the original 3x3 conv (branch2b) with FSAS.
    """

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d", fsas_cfg: dict | None = None):
        if BasicBlock is None:
            raise ImportError("Cannot import BasicBlock. Please check your Ultralytics version/path.")

        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)

        # In original ResNet BasicBlock, branch2b is the 3x3 conv.
        # Here we replace it by FSAS (FFS + RADDC).
        cfg = fsas_cfg or {}
        self.branch2b = FSAS(
            channels=ch_out,
            kernel_size=3,
            stride=1,
            padding=1,
            **cfg,
        )
