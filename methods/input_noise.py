import wandb

from utils import accuracy, gaussian_noise, grad_norm, weight_norm


def input_noise(net, optimizer, criterion, scheduler, loaders, device, epsilon):
    loader = loaders["retain"]
    net.train()
    loss, acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = gaussian_noise(x.clone(), epsilon)
        logits = net(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss += loss.item()
        acc += accuracy(logits.argmax(dim=-1), y)
        wandb.log({"grad_norm": grad_norm(net), "weight_norm": weight_norm(net)})
    loss /= len(loader)
    acc /= len(loader)
    scheduler.step()
    return loss, acc
