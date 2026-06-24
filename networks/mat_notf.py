import numpy as np
import math
import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torch_utils import misc
from torch_utils import persistence
from networks.basic_module import FullyConnectedLayer, Conv2dLayer, MappingNet, MinibatchStdLayer, DisFromRGB, DisBlock, StyleConv, ToRGB, get_style_code, MADFConv


@misc.profiled_function
def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2 ** stage]


@persistence.persistent_class
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(in_features=in_features, out_features=hidden_features, activation='lrelu')
        self.fc2 = FullyConnectedLayer(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


@misc.profiled_function
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@misc.profiled_function
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


@persistence.persistent_class
class Conv2dLayerPartial(nn.Module):
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Width and height of the convolution kernel.
                 bias            = True,         # Apply additive bias before the activation function?
                 activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
                 conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
                 trainable       = True,         # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size, bias, activation, up, down, resample_filter,
                                conv_clamp, trainable)

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size ** 2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(self, x, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding)
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None

@persistence.persistent_class
class MADFFilterGen(nn.Module):
    """Memory-efficient filter generator matching original MADF design"""
    def __init__(self, mask_channels, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        nhidden = 16  # Keep it lightweight like original

        # Key: Apply stride in the mask processing to reduce resolution
        self.mask_conv = nn.Sequential(
            nn.Conv2d(mask_channels, nhidden, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU()
        )

        # Generate filters at the OUTPUT resolution
        filter_dim = in_channels * kernel_size * kernel_size * out_channels
        self.to_filters = nn.Conv2d(nhidden, filter_dim, kernel_size=1)

    def forward(self, mask):
        # Process mask with stride - this reduces resolution!
        mask_features = self.mask_conv(mask)  # [N, 16, H_out, W_out]

        # Generate filters at output resolution
        filters = self.to_filters(mask_features)
        N, C, H, W = filters.shape

        # Reshape for matmul
        filters = filters.view(N, C, H*W).transpose(1, 2)
        filters = filters.view(N, H*W, self.in_channels * self.kernel_size * self.kernel_size,
                               self.out_channels)

        return filters, mask_features

    def basis_weights(self):
        """Reparametrize the linear per-pixel filter generator as 16+1 standard
        conv kernels: y(p) = sum_j mf_j(p) * conv_Bj(x)(p) + conv_b(x)(p).
        Exact identity with the unfold+matmul path, but avoids materializing the
        [N, HW, C_in*k*k, C_out] filter tensor (~3.6 GB/sample at 128/64 res)."""
        k, cin, cout = self.kernel_size, self.in_channels, self.out_channels
        nhidden = self.to_filters.in_channels
        # to_filters.weight: [cin*k*k*cout, nhidden, 1, 1]; flat index = ((c*k+kh)*k+kw)*cout + o
        W1 = self.to_filters.weight.view(cin, k, k, cout, nhidden)
        Wj = W1.permute(4, 3, 0, 1, 2).reshape(nhidden * cout, cin, k, k)
        Wb = self.to_filters.bias.view(cin, k, k, cout).permute(3, 0, 1, 2)
        return torch.cat([Wj, Wb], dim=0)  # [(nhidden+1)*cout, cin, k, k]




@persistence.persistent_class
class MADFLayer(nn.Module):
    """Complete MADF layer matching original design"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, activation='linear', up=1, down=1,
                 mask_channels=None):
        super().__init__()

        if mask_channels is None:
            mask_channels = 16  # Default from original MADF

        self.down = down
        self.stride = down
        self.padding = kernel_size // 2

        # Filter generator with stride
        self.filter_gen = MADFFilterGen(
            mask_channels=mask_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # Dynamic convolution
        self.conv = MADFConv(out_channels, kernel_size, self.stride, self.padding, activation)

    def forward(self, x, mask):
        # Basis-conv reparametrization of the dynamic convolution (exact identity
        # with forward_reference, without the per-pixel filter tensor)
        mask_features = self.filter_gen.mask_conv(mask)  # [N, nhidden, H_out, W_out]
        W_all = self.filter_gen.basis_weights()
        z = F.conv2d(x, W_all, stride=self.stride, padding=self.padding)
        N, _, H_out, W_out = z.shape
        nhidden = mask_features.shape[1]
        cout = self.conv.out_channels
        z = z.view(N, nhidden + 1, cout, H_out, W_out)
        y = (z[:, :nhidden] * mask_features.unsqueeze(2)).sum(dim=1) + z[:, nhidden]
        y = y + self.conv.bias.view(1, -1, 1, 1)
        if self.conv.activation_fn is not None:
            y = self.conv.activation_fn(y)
        return y, mask_features

    def forward_reference(self, x, mask):
        # Original unfold+matmul path, kept for the equality unit test
        filters, mask_features = self.filter_gen(mask)
        x = self.conv(x, filters)
        return x, mask_features

@persistence.persistent_class
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, down_ratio=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.k = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.v = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.proj = FullyConnectedLayer(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask_windows=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1)
        q = self.q(norm_x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(norm_x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        if mask_windows is not None:
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            bias = attn_mask_windows.masked_fill(attn_mask_windows == 0, float(-100.0)).masked_fill(
                attn_mask_windows == 1, float(0.0))
            # windows with no valid token would otherwise degenerate to a uniform
            # average over hole tokens; fall back to content-based attention there
            has_valid = (attn_mask_windows.sum(dim=-1, keepdim=True) > 0).to(bias.dtype)
            attn = attn + bias * has_valid
            with torch.no_grad():
                mask_windows = torch.clamp(torch.sum(mask_windows, dim=1, keepdim=True), 0, 1).repeat(1, N, 1)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows


@persistence.persistent_class
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, down_ratio=1, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    down_ratio=down_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                    proj_drop=drop)

        self.fuse = FullyConnectedLayer(in_features=dim * 2, out_features=dim, activation='lrelu')

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, mask=None):
        # H, W = self.input_resolution
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if mask is not None:
                shifted_mask = torch.roll(mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if mask is not None:
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, mask_windows = self.attn(x_windows, mask_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows, mask_windows = self.attn(x_windows, mask_windows, mask=self.calculate_mask(x_size).to(x.device))  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            if mask is not None:
                mask = torch.roll(shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)

        # FFN
        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        return x, mask


@persistence.persistent_class
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       down=down,
                                       )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistence.persistent_class
class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       up=up,
                                       )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask



@persistence.persistent_class
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, down_ratio=1,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, down_ratio=down_ratio, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.conv = Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, activation='lrelu')

    def forward(self, x, x_size, mask=None):
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask = checkpoint.checkpoint(blk, x, x_size, mask)
            else:
                x, mask = blk(x, x_size, mask)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistence.persistent_class
class ToToken(nn.Module):
    def __init__(self, in_channels=3, dim=128, kernel_size=5, stride=1):
        super().__init__()

        self.proj = Conv2dLayerPartial(in_channels=in_channels, out_channels=dim, kernel_size=kernel_size, activation='lrelu')

    def forward(self, x, mask):
        x, mask = self.proj(x, mask)

        return x, mask

#----------------------------------------------------------------------------

@persistence.persistent_class
class EncFromRGB(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 activation=activation,
                                 )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x

@persistence.persistent_class
class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 down=2,
                                 )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x


@persistence.persistent_class
class Encoder(nn.Module):
    def __init__(self, res_log2, img_channels, activation, patch_size=5, channels=16, drop_path_rate=0.1):
        super().__init__()

        self.resolution = []

        for idx, i in enumerate(range(res_log2, 3, -1)):  # from input size to 16x16
            res = 2 ** i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i+1), nf(i), activation)
            setattr(self, 'EncConv_Block_%dx%d' % (res, res), block)

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, 'EncConv_Block_%dx%d' % (res, res))(x)
            out[res_log2] = x

        return out


@persistence.persistent_class
class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation, drop_rate):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
            Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
            Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(in_features=in_channels,
                                      out_features=out_channels,
                                      activation=activation)
        # self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))
        # x = self.dropout(x)

        return x


@persistence.persistent_class
class DecBlockFirstV2(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )
        self.conv1 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DecBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, ws, gs, E_features, noise_mode='random'):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


@persistence.persistent_class
class Decoder(nn.Module):
    def __init__(self, res_log2, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.Dec_16x16 = DecBlockFirstV2(4, nf(4), nf(4), activation, style_dim, use_noise, demodulate, img_channels)
        for res in range(5, res_log2 + 1):
            setattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res),
                    DecBlock(res, nf(res - 1), nf(res), activation, style_dim, use_noise, demodulate, img_channels))
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img


@persistence.persistent_class
class DecStyleBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2 ** res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2 ** res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    # Added 'mask' to the forward pass signature
    def forward(self, x, img, style, skip, mask, noise_mode='random'):
        x = self.conv0(x, style, noise_mode=noise_mode)

        # Replaced 'x = x + skip' with Teacher-Forcing logic

        # ABLATION (w/o Teacher-Forcing): standard additive skip, no hard gating
        x_fused = x + skip

        x = self.conv1(x_fused, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img


@persistence.persistent_class
class FirstStage(nn.Module):
    def __init__(self, img_channels, img_resolution=256, dim=180, w_dim=512,
                 use_noise=False, demodulate=True, activation='lrelu'):
        super().__init__()
        res = 64
        channel_schedule = [64, 128, 180]

        # This layer processes the image at 256x256 to capture the first skip
        self.conv_first = Conv2dLayer(in_channels=img_channels + 1, out_channels=channel_schedule[0], kernel_size=3,
                                      activation=activation)

        self.enc_conv = nn.ModuleList()

        self.enc_conv.append(
            MADFLayer(
                in_channels=channel_schedule[0],
                out_channels=channel_schedule[0],
                kernel_size=3,
                down=2,
                mask_channels=1,
                activation=activation
            )
        )
        self.enc_conv.append(
            MADFLayer(
                in_channels=channel_schedule[0],
                out_channels=channel_schedule[1],
                kernel_size=3,
                down=2,
                mask_channels=16,
                activation=activation
            )
        )

        self.expand_final = Conv2dLayer(
            in_channels=channel_schedule[1],
            out_channels=dim,
            kernel_size=3,
            activation=activation
        )

        # Renamed bridge layers for clarity
        self.skip_proj_256 = Conv2dLayer(in_channels=channel_schedule[0], out_channels=dim, kernel_size=1,
                                         activation=activation)
        self.skip_proj_128 = Conv2dLayer(in_channels=channel_schedule[0], out_channels=dim, kernel_size=1,
                                         activation=activation)
        self.skip_proj_64 = Conv2dLayer(in_channels=channel_schedule[1], out_channels=dim, kernel_size=1,
                                        activation=activation)

        # Transformer blocks
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1 / 2, 1 / 2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.tran = nn.ModuleList()
        tran_res = res
        for i, depth in enumerate(depths):
            tran_res = int(tran_res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1 / ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.tran.append(
                BasicLayer(dim=dim, input_resolution=[tran_res, tran_res], depth=depth, num_heads=num_heads,
                           window_size=window_sizes[i], drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                           downsample=merge)
            )

        # Style modules and decoder setup
        down_conv = []
        for i in range(int(np.log2(16))):
            down_conv.append(
                Conv2dLayer(in_channels=dim, out_channels=dim, kernel_size=3, down=2, activation=activation))
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)
        self.to_style = FullyConnectedLayer(in_features=dim, out_features=dim * 2, activation=activation)
        self.ws_style = FullyConnectedLayer(in_features=w_dim, out_features=dim, activation=activation)
        mask_style_channels = 64
        self.mask_style_encoder = nn.Sequential(
            Conv2dLayer(1, 16, kernel_size=4, down=2, activation='lrelu'),
            Conv2dLayer(16, 32, kernel_size=4, down=2, activation='lrelu'),
            Conv2dLayer(32, mask_style_channels, kernel_size=4, down=2, activation='lrelu'),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.style_fusion = FullyConnectedLayer(in_features=dim * 2 + dim + mask_style_channels, out_features=dim * 3)
        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        for i in range(down_time):
            block_res = res * (2 ** (i + 1))
            self.dec_conv.append(
                DecStyleBlock(block_res, dim, dim, activation, style_dim, use_noise, demodulate, img_channels)
            )

    def forward(self, images_in, masks_in, ws, noise_mode='random'):
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        encoder_skips = []
        mask = masks_in

        # Process at full 256x256 to get the first skip
        x = self.conv_first(x)
        encoder_skips.append(self.skip_proj_256(x))

        # Downsample with MADF Layer 1 (256 -> 128)
        x, mask_features = self.enc_conv[0](x, mask)
        encoder_skips.append(self.skip_proj_128(x))
        mask = mask_features

        # Downsample with MADF Layer 2 (128 -> 64)
        x, mask_features = self.enc_conv[1](x, mask)
        encoder_skips.append(self.skip_proj_64(x))
        mask = mask_features

        x = self.expand_final(x)

        # Transformer and Style Generation Pass
        x_size = x.size()[-2:]
        x = feature2token(x)
        # Token validity must come from the real hole mask (1=preserved/0=hole);
        # thresholding the unsigned learned MADF features marks hole tokens as
        # valid (inverted), which makes the masked attention suppress known context.
        mask_binary = F.interpolate(masks_in, size=x_size, mode='nearest')
        mask_token = feature2token(mask_binary) if mask_binary is not None else None
        transformer_skips = []
        mid = len(self.tran) // 2
        for i, block in enumerate(self.tran):
            if i < mid:
                x, x_size, mask_token = block(x, x_size, mask_token)
                transformer_skips.append(x)
            elif i > mid:
                x, x_size, mask_token = block(x, x_size, None)
                x = x + transformer_skips[mid - i]
            else:
                x, x_size, mask_token = block(x, x_size, None)
                gs = self.to_style(self.down_conv(token2feature(x, x_size)).flatten(start_dim=1))
                ws_processed = self.ws_style(ws[:, -1])
                ws_processed = F.dropout(ws_processed, p=0.5, training=self.training)
                mask_style = self.mask_style_encoder(masks_in).flatten(start_dim=1)
                combined_styles = torch.cat([gs, ws_processed, mask_style], dim=1)
                style = self.style_fusion(combined_styles)

        # Decoder pass
        x = token2feature(x, x_size).contiguous()
        img = None

        encoder_skips.reverse()  # Now the list is [skip_64, skip_128, skip_256]

        for i, block in enumerate(self.dec_conv):
            # Block 0 (64->128) needs the 128x128 skip, which is encoder_skips[1]
            # Block 1 (128->256) needs the 256x256 skip, which is encoder_skips[2]
            current_skip = encoder_skips[i + 1]
            x, img = block(x, img, style, current_skip, masks_in, noise_mode=noise_mode)

        # Final Ensemble
        img = img * (1 - masks_in) + images_in * masks_in
        return img


@persistence.persistent_class
class SynthesisNet(nn.Module):
    def __init__(self,
                 w_dim,                     # Intermediate latent (W) dimensionality.
                 img_resolution,            # Output image resolution.
                 img_channels   = 3,        # Number of color channels.
                 channel_base   = 32768,    # Overall multiplier for the number of channels.
                 channel_decay  = 1.0,
                 channel_max    = 512,      # Maximum number of channels in any layer.
                 activation     = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 drop_rate      = 0.5,
                 use_noise      = True,
                 demodulate     = True,
                 ):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        # first stage
        self.first_stage = FirstStage(img_channels, img_resolution=img_resolution, w_dim=w_dim, use_noise=False, demodulate=demodulate)

        # second stage
        self.enc = Encoder(resolution_log2, img_channels, activation, patch_size=5, channels=16)
        self.to_square = FullyConnectedLayer(in_features=w_dim, out_features=16*16, activation=activation)
        self.to_style = ToStyle(in_channels=nf(4), out_channels=nf(2) * 2, activation=activation, drop_rate=drop_rate)
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder(resolution_log2, activation, style_dim, use_noise, demodulate, img_channels)

    def forward(self, images_in, masks_in, ws, noise_mode='random', return_stg1=False):
        out_stg1 = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode)

        # encoder
        x = images_in * masks_in + out_stg1 * (1 - masks_in)
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        E_features = self.enc(x)

        fea_16 = E_features[4]
        mul_map_const = getattr(self, 'mul_map_const', None)  # eval-sweep override
        if mul_map_const is None:
            mul_map = torch.ones_like(fea_16) * 0.5
            mul_map = F.dropout(mul_map, training=self.training)
        else:
            mul_map = torch.ones_like(fea_16) * mul_map_const
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(add_n, size=fea_16.size()[-2:], mode='bilinear', align_corners=False)
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        E_features[4] = fea_16

        # style
        gs = self.to_style(fea_16)

        # decoder
        img = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode)

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        if not return_stg1:
            return img
        else:
            return img, out_stg1


@persistence.persistent_class
class Generator(nn.Module):
    def __init__(self,
                 z_dim,                  # Input latent (Z) dimensionality, 0 = no latent.
                 c_dim,                  # Conditioning label (C) dimensionality, 0 = no label.
                 w_dim,                  # Intermediate latent (W) dimensionality.
                 img_resolution,         # resolution of generated image
                 img_channels,           # Number of input color channels.
                 synthesis_kwargs = {},  # Arguments for SynthesisNetwork.
                 mapping_kwargs   = {},  # Arguments for MappingNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = SynthesisNet(w_dim=w_dim,
                                      img_resolution=img_resolution,
                                      img_channels=img_channels,
                                      **synthesis_kwargs)
        self.mapping = MappingNet(z_dim=z_dim,
                                  c_dim=c_dim,
                                  w_dim=w_dim,
                                  num_ws=self.synthesis.num_layers,
                                  **mapping_kwargs)

    def forward(self, images_in, masks_in, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False,
                noise_mode='random', return_stg1=False):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          skip_w_avg_update=skip_w_avg_update)

        if not return_stg1:
            img = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode)
            return img
        else:
            img, out_stg1 = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode, return_stg1=True)
            return img, out_stg1


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,                        # Conditioning label (C) dimensionality.
                 img_resolution,               # Input resolution.
                 img_channels,                 # Number of input color channels.
                 channel_base       = 32768,    # Overall multiplier for the number of channels.
                 channel_max        = 512,      # Maximum number of channels in any layer.
                 channel_decay      = 1,
                 cmap_dim           = None,     # Dimensionality of mapped conditioning label, None = default.
                 activation         = 'lrelu',
                 mbstd_group_size   = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
                 mbstd_num_channels = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4
        self.resolution_log2 = resolution_log2

        if cmap_dim == None:
            cmap_dim = nf(2)
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        if c_dim > 0:
            self.mapping = MappingNet(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None)

        Dis = [DisFromRGB(img_channels+1, nf(resolution_log2), activation)]
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res), nf(res-1), activation))

        if mbstd_num_channels > 0:
            Dis.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
        Dis.append(Conv2dLayer(nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation))
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnectedLayer(nf(2)*4**2, nf(2), activation=activation)
        self.fc1 = FullyConnectedLayer(nf(2), 1 if cmap_dim == 0 else cmap_dim)

        # for 64x64
        Dis_stg1 = [DisFromRGB(img_channels+1, nf(resolution_log2) // 2, activation)]
        for res in range(resolution_log2, 2, -1):
            Dis_stg1.append(DisBlock(nf(res) // 2, nf(res - 1) // 2, activation))

        if mbstd_num_channels > 0:
            Dis_stg1.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
        Dis_stg1.append(Conv2dLayer(nf(2) // 2 + mbstd_num_channels, nf(2) // 2, kernel_size=3, activation=activation))
        self.Dis_stg1 = nn.Sequential(*Dis_stg1)

        self.fc0_stg1 = FullyConnectedLayer(nf(2) // 2 * 4 ** 2, nf(2) // 2, activation=activation)
        self.fc1_stg1 = FullyConnectedLayer(nf(2) // 2, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, images_in, masks_in, images_stg1, c, return_feats=False):
        feats = []
        x = torch.cat([masks_in - 0.5, images_in], dim=1)
        for layer in self.Dis:
            x = layer(x)
            if return_feats and isinstance(layer, DisBlock):
                feats.append(x)
        x = self.fc1(self.fc0(x.flatten(start_dim=1)))

        feats_stg1 = []
        x_stg1 = torch.cat([masks_in - 0.5, images_stg1], dim=1)
        for layer in self.Dis_stg1:
            x_stg1 = layer(x_stg1)
            if return_feats and isinstance(layer, DisBlock):
                feats_stg1.append(x_stg1)
        x_stg1 = self.fc1_stg1(self.fc0_stg1(x_stg1.flatten(start_dim=1)))

        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
            x_stg1 = (x_stg1 * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        if return_feats:
            return x, x_stg1, feats, feats_stg1
        return x, x_stg1


if __name__ == '__main__':
    device = torch.device('cuda:0')
    batch = 1
    res = 512
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3).to(device)
    D = Discriminator(c_dim=0, img_resolution=res, img_channels=3).to(device)
    img = torch.randn(batch, 3, res, res).to(device)
    mask = torch.randn(batch, 1, res, res).to(device)
    z = torch.randn(batch, 512).to(device)
    G.eval()

    # def count(block):
    #     return sum(p.numel() for p in block.parameters()) / 10 ** 6
    # print('Generator', count(G))
    # print('discriminator', count(D))

    with torch.no_grad():
        img, img_stg1 = G(img, mask, z, None, return_stg1=True)
    print('output of G:', img.shape, img_stg1.shape)
    score, score_stg1 = D(img, mask, img_stg1, None)
    print('output of D:', score.shape, score_stg1.shape)
