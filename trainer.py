import torch
import wandb

from utils import grad_norm, weight_norm


def train(net, optimizer, criterion, scheduler, loader, device):
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = net(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss += loss.item()
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    loss /= len(loader)
    scheduler.step()
    return loss, acc


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
