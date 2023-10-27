import wandb

from utils import grad_norm, weight_norm


def optimize(optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def train(net, optimizer, criterion, scheduler, loaders, device):
    loader = loaders["train"]
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss = criterion(logits, y)
        optimize(optimizer, loss)
        loss += loss.item()
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc


def retrain(net, optimizer, criterion, scheduler, loaders, device):
    from torchvision.models import resnet18

    net = resnet18(num_classes=10)
    net.to(device)
    loader = loaders["retain"]
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss = criterion(logits, y)
        optimize(optimizer, loss)
        loss += loss.item()
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc


def finetune(net, optimizer, criterion, scheduler, loaders, device):
    loader = loaders["retain"]
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss = criterion(logits, y)
        optimize(optimizer, loss)
        loss += loss.item()
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc


def gradient_ascent(net, optimizer, criterion, scheduler, loaders, device):
    loader = loaders["forget"]
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss = -criterion(logits, y)
        optimize(optimizer, loss)
        loss += loss.item()
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc


def rand_labels(net, optimizer, criterion, scheduler, loaders, device):
    loader = loaders["rand_labels"]
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss = criterion(logits, y)
        optimize(optimizer, loss)
        loss += loss.item()
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc


def second_best_labels(net, optimizer, criterion, scheduler, loaders, device):
    loader = loaders["second_best_labels"]
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss = criterion(logits, y)
        optimize(optimizer, loss)
        loss += loss.item()
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc
