import random
import time
from copy import deepcopy
from functools import wraps

import numpy as np
import torch


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed_time = self.end - self.start


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


@torch.no_grad()
def evaluate(net, criterion, loader, device):
    net.eval()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss = criterion(logits, y)
        loss += loss.item()
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
    loss /= len(loader)
    acc /= len(loader)
    return loss, acc


def assign_rand_labels(loader):
    forget_labels = torch.cat([y for _, y in loader])
    rand_loader = deepcopy(loader)

    for _, y in rand_loader:
        y[:] = torch.randint_like(y, 0, 10)

    rand_forget_labels = torch.cat([y for _, y in rand_loader])
    acc = (forget_labels == rand_forget_labels).float().mean().item()
    print(f"Random accuracy: {acc:.4f}")
    return rand_loader


def assign_second_best_labels(net, loader, device):
    forget_labels = torch.cat([y for _, y in loader])
    with torch.no_grad():
        net.eval()
        new_loader = deepcopy(loader)
        for x, y in new_loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            _, indices = logits.topk(2, dim=1)
            second_best_labels = indices[:, 1]
            y[:] = second_best_labels
    second_best_forget_labels = torch.cat([y for _, y in new_loader])
    acc = (forget_labels == second_best_forget_labels).float().mean().item()
    print(f"Second best accuracy: {acc:.4f}")
    return new_loader


def time_to_id():
    # Get the current time in nanoseconds
    nanoseconds = int(time.time() * 1e9)
    return str(nanoseconds)


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


@torch.no_grad()
def add_image_noise(x, epsilon):
    noise = torch.normal(0, 1, x.shape).to(x.device) * epsilon
    x.add_(noise)


@torch.no_grad()
def add_grad_noise(grad, epsilon):
    noise = (
        torch.normal(0, 1, grad.shape, dtype=grad.dtype, device=grad.device) * epsilon
    )
    grad.add_(noise)
