# -*- coding: utf-8 -*-
"""
Appendix W: the two standard architectures of the second test bed (unrelated to GAPL), wrapped in the same forward interface as GAPLSegNetV5:
    out = model(query_images, support_images=None, support_masks=None) -> {'logits','base_logits','predictions'}
The support arguments are ignored (single-input segmentation networks; the same structural identity as arms A/B). **Committed before it produced any number.**

  deeplabv3_r50 : torchvision deeplabv3_resnet50, ImageNet-1k backbone (ResNet50_Weights.IMAGENET1K_V1), randomly initialised head, num_classes=1, no auxiliary head
  segformer_b1  : transformers SegformerForSemanticSegmentation, MiT-B1 encoder pre-trained on ImageNet-1k (nvidia/mit-b1), randomly initialised decode head, num_labels=1;
                  logits at 1/4 resolution, bilinearly up-sampled to the input size
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ('deeplabv3_r50', 'segformer_b1')
HF_MIT_B1 = 'nvidia/mit-b1'
# structural constants of MiT-B1 (identical to the nvidia/mit-b1 config; hard-coded so the model can be built offline)
MIT_B1 = dict(num_channels=3, num_encoder_blocks=4, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
              hidden_sizes=[64, 128, 320, 512], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
              num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], decoder_hidden_size=256)


class ArchSeg(nn.Module):
    def __init__(self, arch: str, pretrained: bool = True):
        super().__init__()
        assert arch in ARCHS, arch
        self.arch = arch
        if arch == 'deeplabv3_r50':
            from torchvision.models.segmentation import deeplabv3_resnet50
            from torchvision.models import ResNet50_Weights
            self.net = deeplabv3_resnet50(weights=None,
                                          weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
                                          num_classes=1, aux_loss=False)
        else:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
            if pretrained:
                self.net = SegformerForSemanticSegmentation.from_pretrained(HF_MIT_B1, num_labels=1)
                c = self.net.config
                for k, v in MIT_B1.items():
                    assert list(getattr(c, k)) == v if isinstance(v, list) else getattr(c, k) == v, (k, getattr(c, k), v)
            else:
                self.net = SegformerForSemanticSegmentation(SegformerConfig(num_labels=1, **MIT_B1))

    def forward(self, x, support_images=None, support_masks=None):
        if self.arch == 'deeplabv3_r50':
            logits = self.net(x)['out']
        else:
            logits = self.net(pixel_values=x).logits
            logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return {'logits': logits, 'base_logits': logits, 'predictions': torch.sigmoid(logits)}

    def set_epoch_progress(self, *args, **kwargs):   # called by the GAPL training loop; no-op here
        return None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
