import argparse

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import get_cifar10, prepare_splits
from model import get_cnn, get_resnet18
from trainer import evaluate, train

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="cnn",
    choices=["cnn", "resnet18"],
    help="model (default: cnn)",
)
parser.add_argument(
    "--unlearn_method",
    type=str,
    default="ft",
    choices=["ft", "ga", "rl"],
    help="unlearn method (default: ft)",
)
# parser.add_argument("--seed", type=int, default=2, help="random seed (default: 2)")
parser.add_argument(
    "--lr", type=float, default=1e-1, help="learning rate (default: 1e-1)"
)
parser.add_argument(
    "--epochs", type=int, default=30, help="number of epochs (default: 30)"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size (default: 32)"
)
parser.add_argument(
    "--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)"
)
parser.add_argument(
    "--num_workers", type=int, default=8, help="num workers (default: 8)"
)
parser.add_argument(
    "--wandb",
    type=str,
    default="offline",
    choices=["online", "offline"],
    help="wandb mode (default: offline)",
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(args.seed)
# if DEVICE == "cuda":
#     torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

train_dataset, test_dataset = get_cifar10()
train_dataset, val_dataset, retain_dataset, forget_dataset = prepare_splits(
    train_dataset
)
config = vars(args)
config.update(
    {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "retain_size": len(retain_dataset),
        "forget_size": len(forget_dataset),
        "test_size": len(test_dataset),
    }
)
wandb.init(
    project="cifar10-unlearn", entity="yliu298", config=vars(args), mode=args.wandb
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)
retain_loader = DataLoader(
    retain_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
forget_loader = DataLoader(
    forget_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

if args.model == "cnn":
    net = get_cnn(10)
elif args.model == "resnet18":
    net = get_resnet18(10)
else:
    raise ValueError(f"Unknown model: {args.model}. Please choose from cnn, resnet18.")

net.to(DEVICE)
net.load_state_dict(torch.load("pretrained.pt"))
train_criterion = nn.CrossEntropyLoss()
if args.unlearn_method == "ga":
    train_criterion = lambda logits, y: -nn.CrossEntropyLoss()(logits, y)
test_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Evaluate on all splits before training
_, train_acc = evaluate(net, test_criterion, train_loader, DEVICE)
_, val_acc = evaluate(net, test_criterion, val_loader, DEVICE)
_, test_acc = evaluate(net, test_criterion, test_loader, DEVICE)
_, retain_acc = evaluate(net, test_criterion, retain_loader, DEVICE)
_, forget_acc = evaluate(net, test_criterion, forget_loader, DEVICE)
print(
    {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "retain_acc": retain_acc,
        "forget_acc": forget_acc,
    }
)

train_loader = retain_loader if args.unlearn_method == "ft" else forget_loader
if args.unlearn_method == "rl":
    from copy import deepcopy

    forget_labels = torch.cat([y for _, y in forget_loader])

    rand_forget_loader = deepcopy(forget_loader)
    for x, y in rand_forget_loader:
        y[:] = torch.randint_like(y, 0, 10)

    rand_forget_labels = torch.cat([y for _, y in rand_forget_loader])
    acc = (forget_labels == rand_forget_labels).float().mean().item()
    print(f"Random accuracy: {acc:.4f}")
    train_loader = rand_forget_loader

print(f"train_loader size: {len(train_loader.dataset)}")
for epoch in trange(args.epochs):
    train(net, optimizer, train_criterion, scheduler, train_loader, DEVICE)

    retain_loss, retain_acc = evaluate(net, test_criterion, retain_loader, DEVICE)
    forget_loss, forget_acc = evaluate(net, test_criterion, forget_loader, DEVICE)
    val_loss, val_acc = evaluate(net, test_criterion, val_loader, DEVICE)
    test_loss, test_acc = evaluate(net, test_criterion, test_loader, DEVICE)

    wandb.log(
        {
            "epoch": epoch,
            "retain_loss": retain_loss,
            "retain_acc": retain_acc,
            "forget_loss": forget_loss,
            "forget_acc": forget_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
        }
    )

wandb.finish(quiet=True)
