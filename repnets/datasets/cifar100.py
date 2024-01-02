import pickle

import torchvision
from torch.utils.data import DataLoader

from repnets import constants

TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandAugment(num_ops=9, magnitude=16),
        torchvision.transforms.Resize(constants.IMSIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


import pickle

import torchvision
from torch.utils.data import DataLoader

from repnets import constants

TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(constants.IMSIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def get(train_batchsize, val_batchsize, num_workers):
    def _get_labelmap(file):
        with open(file, "rb") as fo:
            res = pickle.load(fo, encoding="bytes")

        res = {i: t.decode("utf8") for i, t in enumerate(res[b"fine_label_names"])}
        return res

    train_dataset = torchvision.datasets.CIFAR100(
        root="data", download=True, train=True, transform=TRANSFORM
    )

    val_dataset = torchvision.datasets.CIFAR100(
        root="data", download=True, train=False, transform=TRANSFORM
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    labelmap = _get_labelmap(file="./data/cifar-100-python/meta")

    return train_dataloader, test_dataloader, labelmap
