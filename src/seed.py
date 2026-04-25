from __future__ import annotations

import random

import numpy as np
import torch
from transformers import set_seed


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
