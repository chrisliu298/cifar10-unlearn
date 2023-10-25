import argparse
import logging

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import get_cifar10, prepare_splits
from methods import train
from model import get_cnn, get_resnet18
from utils import evaluate, time_to_id

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
    filename=f"{time_to_id()}.log",
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="cnn",
    choices=["cnn", "resnet18"],
    help="model (default: cnn)",
)
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
torch.manual_seed(1337)
if DEVICE == "cuda":
    torch.cuda.manual_seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

train_dataset, test_dataset = get_cifar10()
train_dataset, val_dataset, _, _ = prepare_splits(train_dataset)
config = vars(args)
config.update(
    {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
)
wandb.init(project="cifar10-unlearn", entity="yliu298", config=config, mode=args.wandb)
logging.info(config)
logging.debug(
    {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
)

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

net.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

for epoch in trange(args.epochs):
    train_loss, train_acc = train(
        net, optimizer, criterion, scheduler, train_loader, DEVICE
    )
    val_loss, val_acc = evaluate(net, criterion, val_loader, DEVICE)
    test_loss, test_acc = evaluate(net, criterion, test_loader, DEVICE)
    wandb.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
        }
    )
torch.save(net.state_dict(), f"pretrained.pt")
wandb.save("pretrained.pt")
wandb.finish(quiet=True)
