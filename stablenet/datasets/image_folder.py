from torchvision.datasets import ImageFolder


class ImageFolderWithIdx(ImageFolder):

    def __init__(self, root: str, transform=None, **kwargs):
        super().__init__(root, transform=transform, **kwargs)

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        return img, label, item