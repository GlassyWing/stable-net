import torch
from torch import nn
from torch.optim import Adam, SGD
from tqdm import tqdm
import torch.nn.functional as F

from .losses import cross_covariance_loss
from .utils import fourier_mapping


class Replay:

    def __init__(self, alpha):
        self.alpha = alpha
        self.k = len(self.alpha)
        self.buffer_z = []
        self.buffer_w = []

    def reload(self, z, w):
        # ((k + 1) x B, feat_dim), ((k + 1) x B, feat_dim)
        return (torch.cat(self.buffer_z + [z], dim=0),
                torch.cat(self.buffer_w + [w], dim=0))

    def update(self, z, w):
        if len(self.buffer_w) < self.k:
            self.buffer_z.append(z.detach())
            self.buffer_w.append(w.detach())

        for i in range(len(self.buffer_z)):
            self.buffer_z[i] = self.alpha[i] * self.buffer_z[i] + (1 - self.alpha[i]) * z.detach()
            self.buffer_w[i] = self.alpha[i] * self.buffer_w[i] + (1 - self.alpha[i]) * w.detach()


class StableNetTrainer(nn.Module):

    def __init__(self, model, n, alpha, device="cpu"):
        super().__init__()

        self.w = nn.Parameter(torch.ones(size=(n,), device=device), requires_grad=True)
        self.replay = Replay(alpha=alpha)
        self.k = self.replay.k
        self.model = model
        self.prepare_optims(self.w)
        self.device = device
        self.model = self.model.to(device)

    def prepare_optims(self, w):
        self.w_optim = SGD([w], lr=1e-2)
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

    def fit(self, train_loader, test_loader=None, epochs=5, repeat_num=8, save_path=None):
        for epoch in range(epochs):
            train_bar = tqdm(train_loader)
            self.model.train()
            acc = 0.
            count = 0.
            for img, label, index in train_bar:
                img = img.to(self.device)
                label = label.to(self.device)

                z_l, pred_l = self.model(img, need_feat=True)
                w_l = self.w[index]

                if len(self.replay.buffer_z) >= self.k:

                    # update w_l
                    self.w_optim.zero_grad()

                    for i in range(repeat_num):
                        z_o, w_o = self.replay.reload(z_l.detach(), w_l)
                        sample_loss = cross_covariance_loss(fourier_mapping(z_o), w_o)

                        sample_loss += len(self.w) * F.relu(1 - torch.mean(self.w))
                        if i != repeat_num - 1:
                            sample_loss.backward(retain_graph=True)
                        else:
                            sample_loss.backward()

                    self.w_optim.step()

                # update base model
                self.d_optim.zero_grad()
                ce_loss = F.cross_entropy(pred_l, label, reduction="none") * w_l.detach()
                ce_loss = ce_loss.mean()
                ce_loss.backward()
                self.d_optim.step()

                acc += (pred_l.argmax(dim=1) == label).sum().item()
                count += len(img)
                if len(self.replay.buffer_z) >= self.k:
                    train_bar.set_description(f"[{epoch}/{epochs}] acc: {(acc / count):.2f} sample_loss: {sample_loss.item():.2f} ce_loss: {ce_loss.item():.2f}")
                else:
                    train_bar.set_description(
                        f"[{epoch}/{epochs}] acc: {(acc / count):.2f} ce_loss: {ce_loss.item():.2f}")

                self.replay.update(z_l, w_l)

            if save_path is not None:
                torch.save({
                    "model": self.model.state_dict(),
                    "weight": self.w
                }, save_path)

            if test_loader is not None:
                self.evaluate(test_loader)
