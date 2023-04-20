import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Dict, Optional


class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def seed_everything(seed: int) -> None:
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
        model: torch.Module,
        path: Path,
        save_only_state_dict: bool = True,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        epoch: Optional[int] = None
    ) -> None:

    save_dict = {}

    if save_only_state_dict:
        save_dict["model_state_dict"] = model.state_dict()
    else:
        save_dict["model"] = model

    if optimizer is not None:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        save_dict["scheduler_state_dict"] = scheduler.state_dict()

    if epoch is not None:
        save_dict["epoch"] = epoch

    torch.save(save_dict, path)
