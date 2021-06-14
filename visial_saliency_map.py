import os
import cv2

import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, AblationCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from stablenet.models.model import StableNet



if __name__ == '__main__':
    device = "cpu"
    epochs = 15
    n_cls = 102
    # checkpoint = "checkpoints/caltech.pth"
    checkpoint = "checkpoints/caltech_stable.pth"

    # Prepare dataloader
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    t_ds = ImageFolder("G:/data/101_ObjectCategories", transform=train_transform)

    # Prepare model
    model = StableNet(n_cls)
    model.to(device)
    model.eval()
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])

    for param in model.parameters():
        param.requires_grad = False

    # 467
    idx = 6370
    expect_idx = t_ds[idx][1]
    expect = t_ds.classes[expect_idx]

    img = t_ds[idx][0].unsqueeze(0)
    img.requires_grad_()

    p = model(img)
    p_max_index = p.argmax(dim=1)
    p_max = p[0, p_max_index]
    p_max.backward()
    pred = t_ds.classes[p_max_index[0]]

    saliency, _ = torch.max(img.grad.data.abs(), dim=1)

    saliency = saliency[0].cpu().numpy()
    # saliency = cv2.GaussianBlur(, (3, 3), 1)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for axe in axes:
        axe.axis("off")
    axes[0].imshow(img[0].permute(1, 2, 0).detach().numpy())
    axes[0].title.set_text(f"act: {expect}")
    axes[1].imshow(saliency, cmap=plt.cm.hot)
    axes[1].title.set_text(f"pred: {pred}")


    plt.show()
