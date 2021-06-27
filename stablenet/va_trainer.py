import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from .losses import compute_mmd


class VANetTrainer(nn.Module):

    def __init__(self, model, device="cpu"):
        super().__init__()

        self.model = model
        self.prepare_optims()
        self.device = device
        self.model = self.model.to(device)

    def prepare_optims(self):
        self.d_optim = Adam(self.model.parameters(), lr=2e-4, betas=(0.5, 0.99))

    @torch.no_grad()
    def evaluate(self, test_loader):
        eval_bar = tqdm(test_loader)
        acc = 0.
        count = 0.
        self.model.eval()
        for img, label, index in eval_bar:
            img = img.to(self.device)
            label = label.to(self.device)

            pred = self.model(img)
            pred = pred.argmax(dim=1)
            acc += (pred == label).sum().item()
            count += len(img)

            eval_bar.set_description(f"acc: {(acc / count):.2f}")

    def fit(self, train_loader, test_loader=None, epochs=5, save_path=None):
        for epoch in range(epochs):
            train_bar = tqdm(train_loader)
            self.model.train()
            acc = 0.
            count = 0.
            for img, label, index in train_bar:
                img = img.to(self.device)
                label = label.to(self.device)

                feat, pred_l = self.model(img, True)
                real_sample = torch.randn((150, feat.size(1)), device=feat.device, dtype=feat.dtype)

                kl_loss = 1000 * compute_mmd(feat, real_sample)

                # update base model
                self.d_optim.zero_grad()
                ce_loss = F.cross_entropy(pred_l, label) +  kl_loss
                ce_loss.backward()
                self.d_optim.step()

                acc += (pred_l.argmax(dim=1) == label).sum().item()
                count += len(img)
                train_bar.set_description(f"[{epoch}/{epochs}] acc: {(acc / count):.2f} kl_loss: {kl_loss.item():.4f} ce_loss: {ce_loss.item():.2f}")


            if save_path is not None:
                torch.save({
                    "model": self.model.state_dict(),
                }, save_path)

            if test_loader is not None:
                self.evaluate(test_loader)
