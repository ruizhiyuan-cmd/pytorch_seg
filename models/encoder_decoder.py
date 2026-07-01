"""PyTorch port of tenxnet EncoderDecoderHead (tenxnet/vision/models/encoder_decoder_head.py).

Assembles encoder -> FPN -> SegmentationHead and emits the 9-channel Ridgepath logits at
**input resolution** (the targets are full-resolution).

Resolution note: tenxnet's small encoder has a stride-1 stem, so head fusion to ``level=1``
yields full-res output natively (final resize is a no-op). The drop-in ResNet18 has a stride-2
stem (no level-1 feature), so we fuse to the lowest FPN level and the assembly does ONE final
``F.interpolate`` to the input H x W. ``build_seg_model`` picks the right head level + final
mode per backbone.
"""
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_encoder
from .fpn import FPN
from .seg_head import SegmentationHead


class RidgepathSegModel(nn.Module):
    def __init__(self, encoder_name="resnet18", in_chans=2, num_classes=9,
                 fpn_min_level=2, fpn_max_level=5, fpn_filters=32,
                 head_level=2, head_filters=32, head_num_convs=2,
                 prediction_kernel_size=1, final_upsample_mode="bilinear"):
        super().__init__()
        self.encoder = build_encoder(encoder_name, in_chans)
        self.fpn = FPN(self.encoder.output_specs, fpn_min_level, fpn_max_level, fpn_filters)
        self.head = SegmentationHead(
            num_classes=num_classes, level=head_level, in_channels=fpn_filters,
            num_convs=head_num_convs, num_filters=head_filters,
            prediction_kernel_size=prediction_kernel_size, feature_fusion="pyramid_fusion",
        )
        self.final_upsample_mode = final_upsample_mode

    def forward(self, x):
        in_hw = x.shape[-2:]
        feats = self.encoder.forward_features(x)
        dec = self.fpn(feats)
        out = self.head(dec)
        if out.shape[-2:] != in_hw:
            kw = {} if self.final_upsample_mode == "nearest" else {"align_corners": False}
            out = F.interpolate(out, size=in_hw, mode=self.final_upsample_mode, **kw)
        return out


def build_seg_model(encoder_name="resnet18", in_chans=2, num_classes=9, **overrides):
    """Construct a RidgepathSegModel with backbone-appropriate resolution defaults."""
    defaults = dict(fpn_min_level=2, fpn_max_level=5, fpn_filters=32,
                    head_filters=32, head_num_convs=2, prediction_kernel_size=1)
    if encoder_name == "tenxnet_recipe":
        # faithful test.yaml: FPN includes endpoint "1" (min_level 1, P1-P5); fuse to level 1 (full res).
        defaults.update(fpn_min_level=1, head_level=1, final_upsample_mode="bilinear")
    elif encoder_name == "tenxnet_small":
        # stride-1 stem: level "2" is stride 2, fuse to level 1 -> full res.
        defaults.update(head_level=1, final_upsample_mode="bilinear")
    else:  # resnet18 drop-in: lowest FPN level is /4, resize to input.
        defaults.update(head_level=2, final_upsample_mode="bilinear")
    defaults.update(overrides)
    return RidgepathSegModel(encoder_name=encoder_name, in_chans=in_chans,
                             num_classes=num_classes, **defaults)
