import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from stablenet.datasets import ImageFolderWithIdx
from stablenet.models.model import StableNet
from stablenet.va_trainer import VANetTrainer

if __name__ == '__main__':
    batch_size = 32
    device = "cuda:0"
    save_path = "checkpoints/va_caltech_stable.pth"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=(-1, 1), translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolderWithIdx("G:/data/101_ObjectCategories", transform=transform)
    eval_ds = ImageFolderWithIdx("G:\data\domain_adaptation_images/dslr", transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size, shuffle=False, drop_last=True)

    model = StableNet(101, False)
    trainer = VANetTrainer(model, device=device)

    if os.path.exists(save_path):
        pth = torch.load(save_path, map_location=device)
        trainer.model.load_state_dict(pth["model"])

    trainer.fit(train_loader, None, epochs=100, save_path=save_path)
