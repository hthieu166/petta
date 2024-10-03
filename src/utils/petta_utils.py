"""
Adapted from: https://github.com/mariodoebler/test-time-adaptation/blob/main/classification/models/model.py
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import random
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional

from torch import Tensor
from typing import Tuple
from torchvision import datasets

class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
        std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std
    
class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_files: Sequence[str],
        transform: Optional[Callable] = None
    ):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)

    def build_index(self, label_file):
        with open(label_file, "r") as file:
            tmp_items = [line.strip().split() for line in file if line]

        item_list = []
        for img_file, label in tmp_items:
            img_file = f"{os.sep}".join(img_file.split("/"))
            img_path = os.path.join(self.image_root, img_file)
            domain_name = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain_name))

        return item_list

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, domain, img_path

    def __len__(self):
        return len(self.samples)


class ImageNetXMaskingLayer(torch.nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]

class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

def split_up_model(model, arch_name, dataset_name):
    """
    Split up the model into an encoder and a classifier.
    This is required for methods like RMT and AdaContrast
    :param model: model to be split up
    :param arch_name: name of the network
    :param dataset_name: name of the dataset
    :return: encoder and classifier
    """
    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
        # split up models loaded from timm
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)

    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        if isinstance(model, nn.DataParallel):
            encoder = model.module.encoder
            classifier = model.module.fc
        else:
            encoder = model.encoder
            classifier = model.fc
    elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50", "Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    # add a masking layer to the classifier
    if dataset_name == "imagenet_a":
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(IMAGENET_A_MASK))
    elif dataset_name == "imagenet_r":
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(IMAGENET_R_MASK))
    elif dataset_name == "imagenet_d109":
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(IMAGENET_D109_MASK))

    return encoder, classifier


def complete_data_dir_path(root, dataset_name):
    dataset_name = dataset_name.replace("_recur","")
    # map dataset name to data directory name
    mapping = {"imagenet": "",
               "imagenetc": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_a": "imagenet-a",
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "domainnet126": "", # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "office31": "office-31",
               "visda": "visda-2017",
               "cifar10": "",  # do not change the following values
               "cifar10_c": "",
               "cifar100": "",
               "cifar100_c": "",
               "ccc": "ImageNet-C",
               }
    return os.path.join(root, mapping[dataset_name])


def get_transform(dataset_name, adaptation):
    """
    Get transformation pipeline
    Note that the data normalization is done inside of the model
    :param dataset_name: Name of the dataset
    :param adaptation: Name of the adaptation method
    :return: transforms
    """
    # create non-method specific transformation
    if dataset_name in {"cifar10", "cifar100"}:
        transform = transforms.Compose([transforms.ToTensor()])
    elif dataset_name in {"cifar10c", "cifar100c"}:
        transform = None
    elif dataset_name == "imagenet_c":
        # note that ImageNet-C is already resized and centre cropped
        transform = transforms.Compose([transforms.ToTensor()])
    elif dataset_name in {"domainnet126"}:
        transform = get_augmentation(aug_type="test", res_size=256, crop_size=224)
    else:
        # use classical ImageNet transformation procedure
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    return transform

def get_augmentation(aug_type, res_size=256, crop_size=224):
    if aug_type == "moco-v2":
        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "moco-v2-light":
        transform_list = [
            transforms.Resize((res_size, res_size)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "moco-v1":
        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "plain":
        transform_list = [
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "clip_inference":
        transform_list = [
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    elif aug_type == "test":
        transform_list = [
            transforms.Resize((res_size, res_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    elif aug_type == "imagenet":
        transform_list = [
            transforms.Resize(res_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    else:
        return None

    return transforms.Compose(transform_list)

def get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=True, ckpt_path=None, num_samples=None, percentage=1, workers=4):
    # create the name of the corresponding source dataset
    dataset_name = dataset_name[:-1] if dataset_name in {"cifar10c", "cifar100c", "imagenetc"} else dataset_name

    # complete the root path to the full dataset path
    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)

    # setup the transformation pipeline
    transform = get_transform(dataset_name, adaptation)

    # create the source dataset
    if dataset_name == "cifar10":
        source_dataset = torchvision.datasets.CIFAR10(root=root_dir,
                                                      train=train_split,
                                                      download=True,
                                                      transform=transform)
    elif dataset_name == "cifar100":
        source_dataset = torchvision.datasets.CIFAR100(root=root_dir,
                                                       train=train_split,
                                                       download=True,
                                                       transform=transform)
    elif dataset_name == "imagenet":
        split = "train" if train_split else "val"
        source_dataset = torchvision.datasets.ImageNet(root=data_dir,
                                                       split=split,
                                                       transform=transform)
    elif dataset_name in {"domainnet126", "office31", "visda"}:
        src_domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
        source_data_list = [os.path.join("src/data/lists", f"{dataset_name}_lists", f"{src_domain}_list.txt")]
        source_dataset = ImageList(image_root=data_dir,
                                   label_files=source_data_list,
                                   transform=transform)
        print(f"Loading source data from list: {source_data_list[0]}")
    elif dataset_name in {"imagenet", "ccc"}:
        split = "train" if train_split else "val"
        source_dataset = datasets.ImageNet(data_dir=data_dir,
                                                dataset_name="imagenet_v2",
                                                split=split,
                                                transform=transform)
    else:
        raise ValueError("Dataset not supported.")
    
    if percentage < 1.0 or num_samples:    # reduce the number of source samples
        if dataset_name in {"cifar10", "cifar100"}:
            nr_src_samples = source_dataset.data.shape[0]
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            inds = random.sample(range(0, nr_src_samples), nr_reduced)
            source_dataset.data = source_dataset.data[inds]
            source_dataset.targets = [source_dataset.targets[k] for k in inds]
        else:
            nr_src_samples = len(source_dataset.samples)
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            source_dataset.samples = random.sample(source_dataset.samples, nr_reduced)

        print(f"Number of images in source loader: {nr_reduced}/{nr_src_samples} \t Reduction factor = {nr_reduced / nr_src_samples:.4f}")
    # create the source data loader
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=workers,
                                                drop_last=False)
    print(f"Number of images and batches in source loader: #img = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_dataset, source_loader