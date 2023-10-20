import argparse

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import get_cifar10, prepare_splits
from model import get_cnn, get_resnet18
from utils import grad_norm, weight_norm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="cnn",
    choices=["cnn", "resnet18"],
    help="model (default: cnn)",
)
parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
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

wandb.init(
    project="cifar10-unlearn", entity="yliu298", config=vars(args), mode=args.wandb
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

train_dataset, test_dataset = get_cifar10()
train_dataset, val_dataset, _, _ = prepare_splits(train_dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

if args.model == "cnn":
    net = get_cnn(10)
elif args.model == "resnet18":
    net = get_resnet18(10)
else:
    raise ValueError(f"Unknown model: {args.model}. Please choose from cnn, resnet18.")

net = net.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

for epoch in trange(args.epochs):
    wandb.log({"lr": scheduler.get_last_lr()[0]})

    net.train()
    train_acc_epoch, train_loss_epoch, test_acc_epoch, test_loss_epoch = 0, 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = net(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss_step = loss.item()
        train_acc_step = (logits.argmax(dim=-1) == y).float().mean().item()
        wandb.log(
            {"train_loss_step": train_loss_step, "train_acc_step": train_acc_step}
        )
        train_loss_epoch += train_loss_step
        train_acc_epoch += train_acc_step

        wandb.log({"weight_norm": weight_norm(net)})
        wandb.log({"grad_norm": grad_norm(net)})

    train_loss_epoch /= len(train_loader)
    train_acc_epoch /= len(train_loader)

    net.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = net(x)
            loss = criterion(logits, y)
            test_loss_step = loss.item()
            test_acc_step = (logits.argmax(dim=-1) == y).float().mean().item()
            test_loss_epoch += test_loss_step
            test_acc_epoch += test_acc_step

    test_loss_epoch /= len(test_loader)
    test_acc_epoch /= len(test_loader)

    wandb.log(
        {
            "train_loss_epoch": train_loss_epoch,
            "train_acc_epoch": train_acc_epoch,
            "test_loss_epoch": test_loss_epoch,
            "test_acc_epoch": test_acc_epoch,
        }
    )
    scheduler.step()

torch.save(net.state_dict(), f"pretrained.pt")
wandb.save("pretrained.pt")
wandb.finish(quiet=True)
