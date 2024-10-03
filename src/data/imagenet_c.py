import os
from PIL import Image
from src.data.base_dataset import DatumList, TTADatasetBase
from typing import Sequence, Optional
import torchvision.transforms as transforms
from robustbench.data import CORRUPTIONS
import yaml
import json
from src.data.augmentation import get_augmentation

class ImageNetC(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        self.image_root = cfg.TTA_DATA_DIR
        # Get file name of all images in the test dataset
        with open("robustbench/imagenet_test_image_ids.txt", 'r') as file:
            fnames = [line.strip() for line in file if line]
        # Get mapping from class name to idcs
        with open("robustbench/imagenet_class_to_id_map.json", "r") as f:
            self.class_name_to_id = json.load(f)


        # Using corruptions in a provided order
        if cfg.CORRUPTION.ORDER_FILE is not None:
            with open(cfg.CORRUPTION.ORDER_FILE, 'r') as f:
                recur_corruptions = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            recur_corruptions = {i: all_corruption for i in range(cfg.CORRUPTION.RECUR)}

        if "recur" in cfg.CORRUPTION.DATASET:
            recur = cfg.CORRUPTION.RECUR
            test_domains = recur_corruptions[0] + [test_dm + f"_rep{r}" for r in range(1, recur) for test_dm in recur_corruptions[r]]
        else: 
            test_domains = all_corruption
            recur = 1
        

        self.domain_id_to_name = {i: d_name for i, d_name in enumerate(test_domains)}
        self.domain_name_to_id = {self.domain_id_to_name[i]:i for i in self.domain_id_to_name}
        samples = []
        for r in range(recur):
            samples += self.build_index(fnames=fnames, domains = recur_corruptions[r], severities = all_severity, rep=r)
        data_source = [] 
        for i, smpl in enumerate(samples):
            x, y, d = smpl
            d_id =  self.domain_name_to_id[d]
            data_item = DatumList(x, y, d_id)
            data_source.append(data_item)
        
        super().__init__(cfg, data_source)
        self.to_tensor = get_augmentation(aug_type="test", res_size=224, crop_size=224)

    def build_index(self, fnames, domains, severities, rep=0):
        item_list = []
        for domain_name in domains:
            for severity in severities:
                for img_file in fnames:
                    label = self.class_name_to_id[img_file.split('/')[0]]
                    img_path = os.path.join(self.image_root, domain_name, str(severity), img_file)
                    item_list.append((img_path, int(label),  f"{domain_name}_rep{rep}" if rep != 0 else domain_name))
        return item_list
    
    def load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img

