import os
import logging
from PIL import Image
from src.data.base_dataset import DatumList, TTADatasetBase
from src.data.augmentation import get_augmentation

from torch.utils.data import Dataset
from typing import List, Sequence, Callable, Optional

import yaml

logger = logging.getLogger(__name__)
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

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, domain, img_path

    def __len__(self):
        return len(self.samples)

def load_domainnet126(img_path):
    return Image.open(img_path).convert("RGB")

class DomainNet126(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        dataset_name = "domainnet126"
        src_domain = cfg.CKPT_PATH.replace('.pth', '').split(os.sep)[-1].split('_')[1]
        
        test_domains = get_test_domain_sequence(src_domain)
        data_files = [os.path.join("src/data/lists", f"{dataset_name}_lists", dom_name + "_list.txt") for dom_name in test_domains]
        self.image_root = cfg.TTA_DATA_DIR

        if "recur" in cfg.CORRUPTION.DATASET:
            recur = cfg.CORRUPTION.REcur
            test_domains = test_domains + [test_dm + f"_rep{r}" for r in range(1, recur) for test_dm in test_domains]
        else:
            recur = 1
        
        self.domain_id_to_name = {i: d_name for i, d_name in enumerate(test_domains)}
        self.domain_name_to_id = {self.domain_id_to_name[i]:i for i in self.domain_id_to_name}
        samples = []

        for rep in range(recur):
            for file in data_files:
                samples += self.build_index(label_file=file, rep=rep)
        
        data_source = [] 
        for i, smpl in enumerate(samples):
            x, y, d = smpl
            d_id =  self.domain_name_to_id[d]
            data_item = DatumList(x, y, d_id)
            data_source.append(data_item)

        super().__init__(cfg, data_source)

        self.to_tensor = get_augmentation(aug_type="test", res_size=256, crop_size=224)

    def build_index(self, label_file, rep=0):
        with open(label_file, "r") as file:
            tmp_items = [line.strip().split() for line in file if line]
        item_list = []
        for img_file, label in tmp_items:
            img_file = f"{os.sep}".join(img_file.split("/"))
            img_path = os.path.join(self.image_root, img_file)
            domain_name = img_file.split(os.sep)[0]
            if rep != 0:
                domain_name += f"_rep{rep}"
            item_list.append((img_path, int(label), domain_name))
        return item_list
    
    def load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img

def get_test_domain_sequence(train_domain):
    mapping = {"real": ["clipart", "painting", "sketch"],
            "clipart": ["sketch", "real", "painting"],
            "painting": ["real", "sketch", "clipart"],
            "sketch": ["painting", "clipart", "real"],
            }
    return mapping[train_domain]