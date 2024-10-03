import torch

@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

# The self-training loss is adopted from https://github.com/mariodoebler/test-time-adaptation/blob/main/classification/methods/rmt.py
@torch.jit.script
def self_training(x, x_aug, x_ema):# -> torch.Tensor:
    return - 0.25 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.25 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
           - 0.25 * (x_ema.softmax(1) * x_aug.log_softmax(1)).sum(1) - 0.25 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)