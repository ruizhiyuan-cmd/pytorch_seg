"""Shared layer helpers for the ported seg model.

TF BatchNormalization uses momentum=0.99 (moving-average decay) and epsilon=1e-3. PyTorch's
BatchNorm momentum has the opposite convention (running = (1-m)*old + m*new), so the equivalent
PyTorch momentum is ``1 - 0.99 = 0.01``.
"""
import torch.nn as nn

BN_MOMENTUM = 0.01  # == TF norm_momentum 0.99
BN_EPS = 1e-3       # == TF norm_epsilon 0.001


def bn(num_features: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(num_features, momentum=BN_MOMENTUM, eps=BN_EPS)
