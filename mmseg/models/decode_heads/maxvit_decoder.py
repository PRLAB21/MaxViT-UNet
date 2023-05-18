from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch
import torch.nn as nn
import torchvision.transforms as T

from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath

from ..builder import HEADS
from .decode_head import BaseDecodeHead


def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


class MBConv(nn.Module):
    """ MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf.

        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))

        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).

        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.

        Note: This implementation differs slightly from the original MobileNet implementation!

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
            debug=False
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MBConv, self).__init__()
        self.module_name = '[MBConv]'
        self.debug = debug
        # Save parameter
        self.drop_path_rate: float = drop_path
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1)),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            SqueezeExcite(in_chs=out_channels, rd_ratio=0.25),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        )
        # Make skip path
        self.skip_path = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)) \
                            if (in_channels != out_channels) else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output


def window_partition(
        input: torch.Tensor,
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Window partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)

    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.

    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output


def grid_partition(
        input: torch.Tensor,
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Grid partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)

    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    grid = input.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return grid


def grid_reverse(
        grid: torch.Tensor,
        original_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the grid partition.

    Args:
        Grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height, width, and channels
    (H, W), C = original_size, grid.shape[-1]
    # Compute original batch size
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # Fold grid tensor
    output = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.

    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.

    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    # print(coords.shape)
    coords_flatten = torch.flatten(coords, 1)
    # print(coords_flatten.shape)
    # print(coords_flatten[:, :, None].shape, coords_flatten[:, None, :].shape)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    # print(relative_coords.shape)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    # print(relative_coords.shape)
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    # print(relative_coords.shape)
    return relative_coords.sum(-1)


class RelativeSelfAttention(nn.Module):
    """ Relative Self-Attention similar to Swin V1. Implementation inspired by Timms Swin V1 implementation.

    Args:
        in_channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(RelativeSelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.scale: float = num_heads ** -0.5
        self.attn_area: int = grid_window_size[0] * grid_window_size[1]
        # Init layers
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(grid_window_size[0],
                                                                                    grid_window_size[1]))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.

        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        # print('[self.relative_position_bias_table]', self.relative_position_bias_table.shape)
        # print('[self.relative_position_index]', self.relative_position_index.view(-1).min(), self.relative_position_index.view(-1).max())
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        # print('[relative_position_bias]', relative_position_bias.shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # print('[relative_position_bias]', relative_position_bias.shape)
        return relative_position_bias.unsqueeze(0)

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B_, N, C].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B_, N, C].
        """
        # Get shape of input
        B_, N, C = input.shape
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # print('[qkv]', qkv.shape)
        q, k, v = qkv.unbind(0)
        # Scale query
        q = q * self.scale
        # Compute attention maps
        # print('[self._get_relative_positional_bias()]', self._get_relative_positional_bias().shape)
        attn = self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias())
        # print('[attn]', attn.shape)
        # Map value with attention maps
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class MaxViTTransformerBlock(nn.Module):
    """ MaxViT Transformer block.

        With block partition:
        x ← x + Unblock(RelAttention(Block(LN(x))))
        x ← x + MLP(LN(x))

        With grid partition:
        x ← x + Ungrid(RelAttention(Grid(LN(x))))
        x ← x + MLP(LN(x))

        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.

    Args:
        in_channels (int): Number of input channels.
        partition_function (Callable): Partition function to be utilized (grid or window partition).
        reverse_function (Callable): Reverse function to be utilized  (grid or window reverse).
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
    """

    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            debug=False
    ) -> None:
        """ Constructor method """
        super(MaxViTTransformerBlock, self).__init__()
        self.module_name = '[MaxViTTransformerBlock]'
        self.debug = debug
        # Save parameters
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.grid_window_size: Tuple[int, int] = grid_window_size
        # Init layers
        self.norm_1 = norm_layer(in_channels)
        self.attention = RelativeSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=int(mlp_ratio * in_channels),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)].
        """
        # Save original shape
        B, C, H, W = input.shape
        # Perform partition
        input_partitioned = self.partition_function(input, self.grid_window_size)
        # print(self.module_name + '[forward]', '[input_partitioned]', input_partitioned.shape)
        input_partitioned = input_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)
        # print(self.module_name + '[forward]', '[input_partitioned]', input_partitioned.shape)
        # Perform normalization, attention, and dropout
        output = input_partitioned + self.drop_path(self.attention(self.norm_1(input_partitioned)))
        # Perform normalization, MLP, and dropout
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        # print(self.module_name + '[forward]', '[output]', output.shape)
        # Reverse partition
        output = self.reverse_function(output, (H, W), self.grid_window_size)
        return output


class MaxViTBlock(nn.Module):
    """ MaxViT block composed of MBConv block, Block Attention, and Grid Attention.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm,
            debug=False,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTBlock, self).__init__()
        self.module_name = '[MaxViTBlock]'
        self.debug = debug
        if self.debug: print(self.module_name, '[in_channels]', in_channels)
        if self.debug: print(self.module_name, '[out_channels]', out_channels)
        if self.debug: print(self.module_name, '[num_heads]', num_heads)
        if self.debug: print(self.module_name, '[grid_window_size]', grid_window_size)
        if self.debug: print(self.module_name, '[attn_drop]', attn_drop)
        if self.debug: print(self.module_name, '[drop]', drop)
        if self.debug: print(self.module_name, '[drop_path]', drop_path)
        if self.debug: print(self.module_name, '[mlp_ratio]', mlp_ratio)
        # Init MBConv block
        self.mb_conv = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            debug=debug,
        )
        # Init Block and Grid Transformer
        self.block_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=window_partition,
            reverse_function=window_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer,
            debug=debug,
        )
        self.grid_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=grid_partition,
            reverse_function=grid_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer,
            debug=debug,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2] (downscaling is optional)
        """
        # print(self.module_name + '[forward]', '[input]', input.shape)
        output = self.mb_conv(input)
        # print(self.module_name + '[forward]', '[output][mbconv]', output.shape)
        output = self.block_transformer(output)
        # print(self.module_name + '[forward]', '[output][block_att]', output.shape)
        output = self.grid_transformer(output)
        # print(self.module_name + '[forward]', '[output][grid_att]', output.shape)
        return output


class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_layer: Normalization layer to use.
            Default: nn.BatchNorm2d.
        act_layer (dict | None): Activation layer to use.
            Default: nn.Mish.
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.Mish,
                 kernel_size=4,
                 scale_factor=2):
        super(DeconvModule, self).__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        norm = norm_layer(out_channels)
        activate = act_layer()
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""
        out = self.deconv_upsamping(x)
        return out


class MaxViTStage(nn.Module):
    """ Stage of the MaxViT.

    Args:
        depth (int): Depth of the stage.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
    """

    def __init__(
            self,
            depth: int,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm,
            debug=False,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTStage, self).__init__()
        self.module_name = '[MaxViTStage]'
        self.debug = debug
        if self.debug: print(self.module_name, '[depth]', depth)
        if self.debug: print(self.module_name, '[in_channels]', in_channels)
        if self.debug: print(self.module_name, '[out_channels]', out_channels)
        if self.debug: print(self.module_name, '[num_heads]', num_heads)
        if self.debug: print(self.module_name, '[grid_window_size]', grid_window_size)
        if self.debug: print(self.module_name, '[attn_drop]', attn_drop)
        if self.debug: print(self.module_name, '[drop]', drop)
        if self.debug: print(self.module_name, '[drop_path]', drop_path)
        if self.debug: print(self.module_name, '[mlp_ratio]', mlp_ratio)
        # Deconv for upsampling bottom connection
        self.upsample = DeconvModule(
            in_channels=in_channels,
            out_channels=skip_channels,
            norm_layer=norm_layer,
            act_layer=act_layer)
        # Init blocks
        self.blocks = nn.Sequential(*[
            MaxViTBlock(
                in_channels=2*skip_channels if index == 0 else out_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                grid_window_size=grid_window_size,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_transformer=norm_layer_transformer,
                debug=debug
            )
            for index in range(depth)
        ])

    def forward(self, skip, x):
        """ Forward pass.

        Args:
            skip (torch.Tensor): Input tensor of the shape [B, C_skip, H, W].
            x (torch.Tensor): Input tensor of the shape [B, C_in, H // 2, W/2].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H, W].
        """
        
        if self.debug: print(self.module_name + '[forward]', f'[x]', x.shape)
        x = self.upsample(x)
        if self.debug: print(self.module_name + '[forward]', f'[upsample][x]', x.shape)
        if self.debug: print(self.module_name + '[forward]', f'[skip]', skip.shape)
        output = torch.cat([skip, x], dim=1)
        output = self.blocks(output)

        return output


@HEADS.register_module(name='MaxViTDecoder', force=True)
class MaxViTDecoder(BaseDecodeHead):
    """ 
    
    Hybrid Transformer Decoder based on MaxViTBlock

    Args:
        in_channels (int, optional): Number of input channels to the convolutional stem. Default 3
        depths (Tuple[int, ...], optional): Depth of each network stage. Default (2, 2, 5, 2)
        channels (Tuple[int, ...], optional): Number of channels in each network stage. Default (64, 128, 256, 512)
        num_classes (int, optional): Number of classes to be predicted. Default 1000
        embed_dim (int, optional): Embedding dimension of the convolutional stem. Default 64
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
        global_pool (str, optional): Global polling type to be utilized. Default "avg"
    """

    def __init__(
            self,
            in_channels=[64, 128, 256, 512],
            depths=(2, 2, 2),
            output_size=(256, 256),
            num_heads=32,
            grid_window_size=(8, 8),
            attn_drop=0.,
            drop=0.,
            drop_path=0.,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_transformer=nn.LayerNorm,
            debug=False,
            **kwargs
    ):
        """ Constructor method """
        # Call super constructor
        super(MaxViTDecoder, self).__init__(in_channels=in_channels,
                                            in_index=[0, 1, 2, 3],
                                            input_transform='multiple_select',
                                            **kwargs)

        self.module_name = '[MaxViTDecoder]'
        self.debug = debug
        # Init blocks
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        encoder_stages = len(self.in_channels)
        stages = []
        for index, depth in enumerate(depths):
            stages.append(
                MaxViTStage(
                    depth=depth,
                    in_channels=self.in_channels[encoder_stages - index - 1],
                    skip_channels=self.in_channels[encoder_stages - index - 2],
                    out_channels=self.in_channels[encoder_stages - index - 2],
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_transformer=norm_layer_transformer,
                    debug=debug,
                )
            )
        self.stages = nn.ModuleList(stages)

        self.upsize = T.Resize(size=output_size)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """ Gets the names of parameters to not apply weight decay to.

        Returns:
            nwd (Set[str]): Set of parameter names to not apply weight decay to.
        """
        nwd = set()
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    def _forward_feature(self, inputs):
        """ Forward pass of feature extraction.

        Args:
            input List[torch.Tensor]: List of output tensors from encoder corresponding to different scales of feature maps.

        Returns:
            output (torch.Tensor): Output features of decoder layers.
        """
        encoder_stages = len(inputs)
        x = inputs[-1]
        if self.debug: print('[MaxViT][forward_features]', f'[bottleneck][x]', x.shape)
        for i in range(len(self.stages)):
            x = self.stages[i](inputs[encoder_stages - i - 2], x)
            if self.debug: print('[MaxViT][forward_features]', f'[stage{i}][x]', x.shape)
        output = self.upsize(x)
        return output

    def forward(self, inputs):
        """Forward function."""
        TAG = self.module_name + '[forward]'
        if self.debug: print(TAG, '[stem][output]', len(inputs))
        output = self._forward_feature(inputs)
        if self.debug: print(TAG, '[stem][output]', output.shape)
        output = self.cls_seg(output)
        return output


if __name__ == '__main__':
    model = MaxViTDecoder(
        in_channels=(64, 128, 256, 512),
        depths=(2, 2, 2),
        grid_window_size=(8, 8),
        attn_drop=0.2,
        drop=0.2,
        drop_path=0.2,
        debug=True,
    )
    print(model)
    print(sum(param.numel() for param in model.parameters()) / 1e6)
    inputs = [torch.rand(1, 64, 64, 64), torch.rand(1, 128, 32, 32), torch.rand(1, 256, 16, 16), torch.rand(1, 512, 8, 8)]
    output = model(inputs)
    print(output.shape)
