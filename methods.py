import wandb

from utils import accuracy, add_grad_noise, add_image_noise, grad_norm, weight_norm


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
        acc += accuracy(logits.argmax(dim=-1), y)
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
        acc += accuracy(logits.argmax(dim=-1), y)
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
        acc += accuracy(logits.argmax(dim=-1), y)
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
        acc += accuracy(logits.argmax(dim=-1), y)
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
        acc += accuracy(logits.argmax(dim=-1), y)
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
        acc += accuracy(logits.argmax(dim=-1), y)
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc


def image_noise(net, optimizer, criterion, scheduler, loaders, device, epsilon):
    loader = loaders["retain"]
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        add_image_noise(x, epsilon)
        logits = net(x)
        loss = criterion(logits, y)
        optimize(optimizer, loss)
        loss += loss.item()
        acc += accuracy(logits.argmax(dim=-1), y)
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc


def grad_noise(net, optimizer, criterion, scheduler, loaders, device, epsilon):
    loader = loaders["retain"]
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()

        for param in net.parameters():
            if hasattr(param, "grad") and param.grad is not None:
                add_grad_noise(param.grad, epsilon)

        optimizer.step()
        loss += loss.item()
        acc += accuracy(logits.argmax(dim=-1), y)
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc
