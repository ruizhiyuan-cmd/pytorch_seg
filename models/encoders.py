"""Swappable feature-pyramid encoders for the ported seg model.

Both expose ``forward_features(x) -> {"2":C2,"3":C3,"4":C4,"5":C5}`` (NCHW) and an
``output_specs`` dict ({level: channels}) so the FPN is built backbone-agnostically.

  * ``resnet18``  -- the DINO drop-in: torchvision ResNet18 with conv1 -> in_chans, tapping
    layer1..layer4 (C2-C5 = 64/128/256/512 @ stride 4/8/16/32). Identical construction to
    ssl_dino/resnet4ch.py:build_resnet18_4ch, so the DINO checkpoint loads with a plain
    state_dict copy (see restore.py). Stem stride-2.
  * ``tenxnet_small`` -- faithful port of tenxnet/vision/models/encoders/resnet.py (ResNet
    class, stem_type v0): 7x7 stride-1 stem + 3x3 stride-2 maxpool, then 4 basic-block groups
    (start_filter 16, num_filters [16,32,64,128], block_repeats [2,2,2,2]); endpoints "2".."5"
    at stride 2/4/8/16, channels 16/32/64/128. Basic block mirrors nn_blocks.ResidualBlock
    (projection conv1x1+BN on the first block of each group). SE / resnetd / stochastic-depth
    are unused in the ridgepath config and omitted.
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .layers import bn, make_gn


# --------------------------------------------------------------------------- resnet18 drop-in
def _build_resnet18(in_chans: int):
    """Mirror of ssl_dino/resnet4ch.py:build_resnet18_4ch (conv1 -> in_chans, else stock)."""
    model = tv_models.resnet18(weights=None)
    old = model.conv1  # Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
    model.conv1 = nn.Conv2d(in_chans, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=(old.bias is not None))
    return model


class ResNet18Features(nn.Module):
    def __init__(self, in_chans: int = 2):
        super().__init__()
        self.backbone = _build_resnet18(in_chans)
        # avgpool/fc are unused by feature extraction; drop fc so it carries no dead params
        # (mirrors how DINO's MultiCropWrapper sets fc=Identity).
        self.backbone.fc = nn.Identity()
        self.output_specs = {"2": 64, "3": 128, "4": 256, "5": 512}

    def forward_features(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        c2 = b.layer1(x)
        c3 = b.layer2(c2)
        c4 = b.layer3(c3)
        c5 = b.layer4(c4)
        return {"2": c2, "3": c3, "4": c4, "5": c5}

    forward = forward_features


# --------------------------------------------------------------------------- faithful tenxnet small
class DropPath(nn.Module):
    """Stochastic depth (== nn_layers.StochasticDepth): drop the residual branch per-sample
    with prob ``drop_rate`` during training, scaling kept samples by 1/keep_prob."""

    def __init__(self, drop_rate: float = 0.0):
        super().__init__()
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        if self.drop_rate == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_rate
        mask = x.new_empty((x.shape[0], 1, 1, 1)).bernoulli_(keep)
        return x / keep * mask


class BasicBlock(nn.Module):
    """== nn_blocks.ResidualBlock (no SE / resnetd). Optional stochastic depth on the residual."""

    def __init__(self, in_ch: int, filters: int, stride: int, use_projection: bool,
                 drop_path_rate: float = 0.0, norm_layer=None):
        super().__init__()
        nl = norm_layer or bn  # None -> BatchNorm (default path unchanged)
        self.use_projection = use_projection
        if use_projection:
            self.shortcut = nn.Conv2d(in_ch, filters, kernel_size=1, stride=stride, bias=False)
            self.norm0 = nl(filters)
        self.conv1 = nn.Conv2d(in_ch, filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nl(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nl(filters)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.norm0(self.shortcut(x))
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.drop_path(out)  # stochastic depth on residual, before add (matches tenxnet)
        return F.relu(out + shortcut)


class TenxnetSmallResNet(nn.Module):
    """Faithful tenxnet ResNet (encoders/resnet.py).

    ``stem_type`` v0 = single 7x7 stride-1 conv; v1 = three 3x3 stride-1 convs. ``expose_stem``
    adds endpoint "1" (stem output, full res). ``stochastic_depth_rate`` matches tenxnet's
    per-group schedule rate = init * (i+2) / (num_lvls+1) (same rate for all blocks in a group).
    """

    def __init__(self, in_chans: int = 2, start_filter: int = 16,
                 num_filters=(16, 32, 64, 128), block_repeats=(2, 2, 2, 2),
                 stem_type: str = "v0", stochastic_depth_rate: float = 0.0,
                 expose_stem: bool = False, norm_layer=None):
        super().__init__()
        self.expose_stem = expose_stem
        nl = norm_layer or bn  # None -> BatchNorm (default path unchanged)

        def cbr(cin, k):
            return nn.Sequential(
                nn.Conv2d(cin, start_filter, kernel_size=k, stride=1, padding=k // 2, bias=False),
                nl(start_filter), nn.ReLU(inplace=True))

        if stem_type == "v0":
            self.stem = cbr(in_chans, 7)
        elif stem_type == "v1":
            self.stem = nn.Sequential(cbr(in_chans, 3), cbr(start_filter, 3), cbr(start_filter, 3))
        else:
            raise ValueError(f"stem_type must be 'v0' or 'v1', got {stem_type!r}")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_lvls = len(block_repeats)
        self.groups = nn.ModuleList()
        self.output_specs = {}
        if expose_stem:
            self.output_specs["1"] = start_filter
        in_ch = start_filter
        for i, (f, r) in enumerate(zip(num_filters, block_repeats)):
            stride = 1 if i == 0 else 2
            dp = stochastic_depth_rate * (i + 2) / (num_lvls + 1) if stochastic_depth_rate else 0.0
            blocks = [BasicBlock(in_ch, f, stride, use_projection=True, drop_path_rate=dp,
                                 norm_layer=norm_layer)]
            blocks += [BasicBlock(f, f, 1, use_projection=False, drop_path_rate=dp,
                                  norm_layer=norm_layer)
                       for _ in range(1, r)]
            self.groups.append(nn.Sequential(*blocks))
            in_ch = f
            self.output_specs[str(i + 2)] = f

    def forward_features(self, x):
        x = self.stem(x)
        out = {"1": x} if self.expose_stem else {}
        x = self.maxpool(x)
        for i, group in enumerate(self.groups):
            x = group(x)
            out[str(i + 2)] = x
        return out

    forward = forward_features


def build_encoder(name: str, in_chans: int, encoder_norm: str = "bn", gn_max_groups: int = 32):
    """Build a feature-pyramid encoder.

    ``encoder_norm='bn'`` (default) keeps the original BatchNorm path exactly. ``'gn'`` swaps every
    encoder norm to adaptive GroupNorm (via :func:`make_gn`) -- used for the iBOT/DINOv2 SSL encoder
    so it (a) has no batch-coupled running stats and (b) matches the GN SSL backbone key-for-key.
    Only the tenxnet encoders support GN (resnet18 stays BN; it is not used for the SSL->seg handoff).
    """
    if encoder_norm not in ("bn", "gn"):
        raise ValueError(f"encoder_norm must be 'bn' or 'gn', got {encoder_norm!r}")
    nl = make_gn(gn_max_groups) if encoder_norm == "gn" else None
    if name == "resnet18":
        if encoder_norm == "gn":
            raise ValueError("encoder_norm='gn' is not supported for resnet18 (BN-only drop-in)")
        return ResNet18Features(in_chans)
    if name == "tenxnet_small":  # v0 stem, no stochastic depth, endpoints 2..5 (FPN min_level 2)
        return TenxnetSmallResNet(in_chans, norm_layer=nl)
    if name == "tenxnet_recipe":  # faithful test.yaml: v1 stem, SD 0.5, endpoints 1..5 (FPN min_level 1)
        return TenxnetSmallResNet(in_chans, stem_type="v1", stochastic_depth_rate=0.5,
                                  expose_stem=True, norm_layer=nl)
    raise ValueError(f"unknown encoder {name!r} "
                     f"(expected 'resnet18', 'tenxnet_small', or 'tenxnet_recipe')")
