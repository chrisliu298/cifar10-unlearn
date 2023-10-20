import random
from functools import wraps

import numpy as np
import torch


class ControlledSeed:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.pytorch_state = torch.random.get_rng_state()
        self.numpy_state = np.random.get_state()
        self.python_state = random.getstate()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __exit__(self, type, value, traceback):
        torch.random.set_rng_state(self.pytorch_state)
        np.random.set_state(self.numpy_state)
        random.setstate(self.python_state)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


@torch.no_grad()
def weight_norm(net):
    params = torch.cat([p.view(-1) for p in net.parameters() if p.requires_grad])
    return torch.norm(params, p=2).item()


@torch.no_grad()
def grad_norm(net):
    grads = torch.cat([p.grad.view(-1) for p in net.parameters() if p.grad is not None])
    return torch.norm(grads, p=2).item()
