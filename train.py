from torch.utils.data import DataLoader
from torchvision import transforms

from stablenet.datasets import ImageFolderWithIdx
from stablenet.models.model import StableNet
from stablenet.trainer import StableNetTrainer

import numpy as np
if __name__ == '__main__':
    batch_size = 16
    k = 5
    device = "cuda:0"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine((-3, 3), translate=(0.1, 0.1)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolderWithIdx("G:\data\domain_adaptation_images/webcam", transform=transform)
    eval_ds = ImageFolderWithIdx("G:\data\domain_adaptation_images/dslr", transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size, shuffle=False, drop_last=True)

    model = StableNet(31)
    trainer = StableNetTrainer(model, len(train_ds), alpha=np.random.rand(4), device=device)

    trainer.fit(train_loader, eval_loader, epochs=10, repeat_num=3, save_path="checkpoints/office_stable.pth")
