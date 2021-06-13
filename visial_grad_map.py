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


def remove_margin():
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


if __name__ == '__main__':
    device = "cpu"
    epochs = 15
    n_cls = 31
    checkpoint = "checkpoints/office.pth"

    # Prepare dataloader
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    t_ds = ImageFolder("G:\data\domain_adaptation_images/dslr", transform=train_transform)

    # Prepare model
    model = StableNet(31)
    model.to(device)
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])

    cam = GradCAM(model=model, target_layer=model.fe.layer3[-2], use_cuda=False)

    # 467
    idx = 65
    expect_idx = t_ds[idx][1]
    expect = t_ds.classes[expect_idx]
    with torch.no_grad():
        p = model(t_ds[idx][0].unsqueeze(0))
        p_s = torch.softmax(p, dim=-1)
        p = p.argmax(dim=1)

    real_cam = cam(input_tensor=t_ds[idx][0].unsqueeze(0), target_category=expect_idx)
    pred_cam = cam(input_tensor=t_ds[idx][0].unsqueeze(0), target_category=None)
    real_cam = real_cam[0, :]
    pred_cam = pred_cam[0, :]
    visualization_pred = show_cam_on_image(t_ds[idx][0].permute(1, 2, 0).cpu().numpy(), pred_cam)
    visualization_real = show_cam_on_image(t_ds[idx][0].permute(1, 2, 0).cpu().numpy(), real_cam)
    visualization_pred = cv2.cvtColor(visualization_pred, cv2.COLOR_BGR2RGB)
    visualization_real = cv2.cvtColor(visualization_real, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(ncols=2)
    remove_margin()
    for ax in axes.flat:
        # Remove all xticks and yticks...
        ax.set(xticks=[], yticks=[])

    axes[0].imshow(visualization_pred)
    axes[0].title.set_text(f"pred: {t_ds.classes[p[0].item()]} {p_s[0][p[0].item()]:.4f}")

    axes[1].imshow(visualization_real)
    axes[1].title.set_text(f"real: {expect} {p_s[0][expect_idx]:.4f}")

    plt.show()
