from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ImageFolderWithIdx(ImageFolder):

    def __init__(self, root: str, transform=None, **kwargs):
        super().__init__(root, transform=transform, **kwargs)

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        return img, label, item


class ImageLabelWithIdx(Dataset):

    def __init__(self, images, labels, transform=None):
        self.classes = sorted(list(set(labels)))
        self.class_to_idx = {clz: self.classes.index(clz) for clz in self.classes}
        self.samples = list(zip(images, labels))
        self.transform = transform

    def __getitem__(self, item):
        img_path, label = self.samples[item]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[label], item

    def __len__(self):
        return len(self.samples)
