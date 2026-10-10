"""Learning-rate schedules shared by the trainers."""

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup, then linear decay to 0 -- standard transformer
    training schedule. last_epoch lets a resumed run continue the
    schedule from where it left off instead of restarting warmup."""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        remaining = num_training_steps - step
        total_decay_steps = max(1, num_training_steps - num_warmup_steps)
        return max(0.0, remaining / total_decay_steps)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
