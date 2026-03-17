import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


def icnr_linear_(linear: nn.Linear, out_channels: int, scale: int) -> None:
    with torch.no_grad():
        base = torch.empty(linear.in_features, out_channels)
        nn.init.kaiming_normal_(base)
        w = base.repeat(1, scale ** 2).contiguous()
        linear.weight.copy_(w.t())
        if linear.bias is not None:
            linear.bias.zero_()


class AntiAliasBlur(nn.Module):
    def __init__(self, channels: int, kernel: Optional[torch.Tensor] = None):
        super(AntiAliasBlur, self).__init__()
        if kernel is None:
            k = torch.tensor([[1., 2., 1.],
                              [2., 4., 2.],
                              [1., 2., 1.]], dtype=torch.float32) / 16.0
        else:
            k = kernel.to(torch.float32)
        k = k[None, None, :, :]
        self.register_buffer('w', k.repeat(channels, 1, 1, 1), persistent=False)
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.w, bias=None, stride=1, padding=1, groups=self.groups)


def shift_average_4(x: torch.Tensor, data_format: str = 'bhwc') -> torch.Tensor:
    if data_format == 'bhwc':
        x = rearrange(x, 'b h w c -> b c h w')
        fmt = 'bhwc'
    else:
        fmt = 'bchw'

    y = (x +
         x.roll(shifts=(0, 1), dims=(2, 3)) +
         x.roll(shifts=(1, 0), dims=(2, 3)) +
         x.roll(shifts=(1, 1), dims=(2, 3))) / 4.0

    if fmt == 'bhwc':
        y = rearrange(y, 'b c h w -> b h w c')
    return y


class _AntiCheckerboardPixelShuffle(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int,
                 use_post_blur: bool = True, use_shiftavg: bool = False,
                 return_format: str = 'bhwc'):
        super(_AntiCheckerboardPixelShuffle, self).__init__()
        assert scale in (2, 4)
        self.scale = scale
        self.return_format = return_format
        self.use_post_blur = use_post_blur
        self.use_shiftavg = use_shiftavg

        self.expand = nn.Linear(in_ch, out_ch * (scale ** 2), bias=True)
        icnr_linear_(self.expand, out_channels=out_ch, scale=scale)

        self.post_blur = AntiAliasBlur(out_ch) if use_post_blur else nn.Identity()
        self._printed_expand_msg = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Input must be 4D tensor")
        if x.shape[-1] < 8 and x.shape[1] >= 8:  
            x = rearrange(x, 'b c h w -> b h w c')

        B, H, W, Cin = x.shape
        r = self.scale

        if Cin != self.expand.in_features:
            new_expand = nn.Linear(Cin, self.expand.out_features, bias=True).to(x.device)
            with torch.no_grad():
                icnr_linear_(new_expand,
                             out_channels=self.expand.out_features // (self.scale ** 2),
                             scale=self.scale)
            self.expand = new_expand
            if not self._printed_expand_msg:
                print(f"[anti_checkerboard_up] Auto-adjusted expand layer: in_ch {Cin}")
                self._printed_expand_msg = True

        x = self.expand(x)  # [B,H,W,r^2*Cout]

        x = rearrange(x, 'b h w (r1 r2 c) -> b c (h r1) (w r2)', r1=r, r2=r)

        x = self.post_blur(x)

        if self.use_shiftavg:
            x = (x + x.roll((0, 1), (2, 3))
                   + x.roll((1, 0), (2, 3))
                   + x.roll((1, 1), (2, 3))) / 4.0

        if self.return_format == 'bhwc':
            x = rearrange(x, 'b c h w -> b h w c')
        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim: int = None, dim_scale: int = 2, out_dim: Optional[int] = None,
                 in_ch: Optional[int] = None, out_ch: Optional[int] = None, scale: Optional[int] = None,
                 use_post_blur: bool = True, use_shiftavg: bool = False,
                 return_format: str = 'bhwc', **kwargs):
        super(PatchExpand2D, self).__init__()
        if in_ch is None: in_ch = dim
        if scale is None: scale = dim_scale
        if out_ch is None:
            out_ch = out_dim if out_dim is not None else in_ch
        self.core = _AntiCheckerboardPixelShuffle(
            in_ch=in_ch, out_ch=out_ch, scale=scale,
            use_post_blur=use_post_blur, use_shiftavg=use_shiftavg,
            return_format=return_format
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim: int = None, dim_scale: int = 4, out_dim: Optional[int] = None,
                 in_ch: Optional[int] = None, out_ch: Optional[int] = None, scale: Optional[int] = None,
                 use_post_blur: bool = True, use_shiftavg: bool = False,
                 return_format: str = 'bhwc',
                 reduce_to: int = 24, **kwargs):
        super(Final_PatchExpand2D, self).__init__()
        if in_ch is None: in_ch = dim
        if scale is None: scale = dim_scale
        if out_ch is None:
            out_ch = out_dim if out_dim is not None else 1
        self.core = _AntiCheckerboardPixelShuffle(
            in_ch=in_ch, out_ch=out_ch, scale=scale,
            use_post_blur=use_post_blur, use_shiftavg=use_shiftavg,
            return_format=return_format  
        )

        self.return_format = return_format
        self.reduce_to = reduce_to
        self.reduce = None 
        self._printed_reduce_msg = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.core(x)   # [B,H,W,C]
        C = x.shape[-1]

        if C != self.reduce_to:
            if self.reduce is None or (isinstance(self.reduce, nn.Conv2d) and self.reduce.in_channels != C):
                self.reduce = nn.Conv2d(C, self.reduce_to, kernel_size=1)
                self.reduce.to(x.device)
                if not self._printed_reduce_msg:
                    print(f"[Final_PatchExpand2D] Auto-reduce channels {C} -> {self.reduce_to}")
                    self._printed_reduce_msg = True

            x = rearrange(x, 'b h w c -> b c h w')
            x = self.reduce(x)
            x = rearrange(x, 'b c h w -> b h w c')  

        return x

