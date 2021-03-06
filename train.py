import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from stablenet.datasets import ImageFolderWithIdx
from stablenet.models.model import StableNet
from stablenet.trainer import StableNetTrainer

if __name__ == '__main__':
    batch_size = 32
    k = 7
    device = "cuda:0"
    save_path = "checkpoints/caltech_stable.pth"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomAffine(degrees=(-1, 1), translate=(0.05, 0.05)),
        transforms.RandomPerspective(0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolderWithIdx("G:\data/101_ObjectCategories", transform=transform)
    eval_ds = ImageFolderWithIdx("G:\data\domain_adaptation_images/dslr", transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size, shuffle=False, drop_last=True)

    model = StableNet(101, False)
    trainer = StableNetTrainer(model, len(train_ds), alpha=np.random.rand(k), device=device)

    if os.path.exists(save_path):
        pth = torch.load(save_path, map_location=device)
        trainer.w = pth["weight"]
        trainer.model.load_state_dict(pth["model"])

    trainer.fit(train_loader, None, epochs=13, repeat_num=1, save_path=save_path)
