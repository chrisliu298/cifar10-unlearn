import argparse
import logging
import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import get_cifar10, prepare_splits
from methods import (
    finetune,
    grad_noise,
    gradient_ascent,
    input_noise,
    rand_labels,
    retrain,
    second_best_labels,
)
from model import get_cnn, get_resnet18
from utils import (
    Timer,
    assign_rand_labels,
    assign_second_best_labels,
    evaluate,
    time_to_id,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["cnn", "resnet18"],
    required=True,
    help="model (default: cnn)",
)
parser.add_argument(
    "--unlearn_method",
    type=str,
    choices=[
        "retrain",
        "finetune",
        "gradient_ascent",
        "rand_labels",
        "second_best_labels",
        "input_noise",
        "grad_noise",
    ],
    required=True,
    help="unlearn method (default: finetune)",
)
parser.add_argument(
    "--lr", type=float, default=1e-1, help="learning rate (default: 1e-1)"
)
parser.add_argument(
    "--epochs", type=int, default=10, help="number of epochs (default: 10)"
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="batch size (default: 128)"
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
parser.add_argument(
    "--num_forget",
    type=int,
    default=5000,
    help="number of forget samples (default: 5000)",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=None,
    help="epsilon for input/grad noise (default: None)",
)
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
    filename=f"{args.unlearn_method}_{time_to_id()}.log",
)

if args.unlearn_method in ["input_noise", "grad_noise"]:
    assert args.epsilon is not None, "epsilon must be specified for input/grad noise"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

train_dataset, test_dataset = get_cifar10()
num_val, num_forget = 5000, args.num_forget
num_retain = len(train_dataset) - num_val - num_forget
train_dataset, val_dataset, retain_dataset, forget_dataset = prepare_splits(
    train_dataset, [num_val, num_retain, num_forget]
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
wandb.init(project="cifar10-unlearn", entity="yliu298", config=config, mode=args.wandb)
logging.info(config)

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

loaders = {
    "train": train_loader,
    "val": val_loader,
    "test": test_loader,
    "retain": retain_loader,
    "forget": forget_loader,
}

if args.model == "cnn":
    net = get_cnn(10)
elif args.model == "resnet18":
    net = get_resnet18(10)
else:
    raise ValueError(f"Unknown model: {args.model}. Please choose from cnn, resnet18.")

net.to(DEVICE)
if args.unlearn_method != "retrain":
    assert os.path.exists("pretrained.pt"), "pretrained.pt does not exist!"
    net.load_state_dict(torch.load("pretrained.pt"))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Evaluate on all splits before training
_, train_acc = evaluate(net, criterion, loaders["train"], DEVICE)
_, val_acc = evaluate(net, criterion, loaders["val"], DEVICE)
_, test_acc = evaluate(net, criterion, loaders["test"], DEVICE)
_, retain_acc = evaluate(net, criterion, loaders["retain"], DEVICE)
_, forget_acc = evaluate(net, criterion, loaders["forget"], DEVICE)
logging.debug(
    {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "retain_acc": retain_acc,
        "forget_acc": forget_acc,
    }
)

if args.unlearn_method == "rand_labels":
    rand_forget_loader = assign_rand_labels(forget_loader)
    loaders["rand_forget"] = rand_forget_loader
    logging.debug(next(iter(forget_loader))[1])
    logging.debug(next(iter(rand_forget_loader))[1])
elif args.unlearn_method == "second_best_labels":
    second_best_forget_loader = assign_second_best_labels(net, forget_loader, DEVICE)
    loaders["second_best_forget"] = second_best_forget_loader
    logging.debug(next(iter(forget_loader))[1])
    logging.debug(next(iter(second_best_forget_loader))[1])

time_spent = 0.0
for epoch in trange(args.epochs):
    with Timer() as t:
        if args.unlearn_method == "retrain":
            retrain(net, optimizer, criterion, scheduler, loaders, DEVICE)
        elif args.unlearn_method == "finetune":
            finetune(net, optimizer, criterion, scheduler, loaders, DEVICE)
        elif args.unlearn_method == "gradient_ascent":
            gradient_ascent(net, optimizer, criterion, scheduler, loaders, DEVICE)
        elif args.unlearn_method == "rand_labels":
            rand_labels(net, optimizer, criterion, scheduler, loaders, DEVICE)
        elif args.unlearn_method == "second_best_labels":
            second_best_labels(net, optimizer, criterion, scheduler, loaders, DEVICE)
        elif args.unlearn_method == "input_noise":
            input_noise(
                net, optimizer, criterion, scheduler, loaders, DEVICE, args.epsilon
            )
        elif args.unlearn_method == "grad_noise":
            grad_noise(
                net, optimizer, criterion, scheduler, loaders, DEVICE, args.epsilon
            )

    time_spent += t.elapsed_time
    retain_loss, retain_acc = evaluate(net, criterion, retain_loader, DEVICE)
    forget_loss, forget_acc = evaluate(net, criterion, forget_loader, DEVICE)
    val_loss, val_acc = evaluate(net, criterion, val_loader, DEVICE)
    test_loss, test_acc = evaluate(net, criterion, test_loader, DEVICE)

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
wandb.log({"time_spent": time_spent})
wandb.finish(quiet=True)
