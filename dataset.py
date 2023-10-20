import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from utils import ControlledSeed


def get_cifar10():
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=t
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=t
    )
    return train_dataset, test_dataset


@ControlledSeed(42)
def prepare_splits(train_dataset):
    dataset_len = len(train_dataset)
    rand_indices = torch.randperm(dataset_len)

    num_val, num_retain, num_forget = 5000, 40000, 5000
    val_indices = rand_indices[:num_val]
    retain_indices = rand_indices[num_val : num_val + num_retain]
    forget_indices = rand_indices[num_val + num_retain :]
    train_indices = torch.cat([retain_indices, forget_indices])

    # Make sure no overlap between retain and forget
    assert len(set(retain_indices) & set(forget_indices)) == 0
    # Make sure all retain are in train
    assert len(set(retain_indices) - set(train_indices)) == 0
    # Make sure all forget are in train
    assert len(set(forget_indices) - set(train_indices)) == 0

    train_dataset_split = Subset(train_dataset, train_indices)
    val_dataset = Subset(train_dataset, val_indices)
    retain_dataset = Subset(train_dataset, retain_indices)
    forget_dataset = Subset(train_dataset, forget_indices)

    print(
        {
            "train": len(train_dataset_split),
            "val": len(val_dataset),
            "retain": len(retain_dataset),
            "forget": len(forget_dataset),
        }
    )

    return train_dataset_split, val_dataset, retain_dataset, forget_dataset
