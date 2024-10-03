from src.data.base_dataset import TTADatasetBase, DatumRaw
from robustbench.data import load_cifar10c, load_cifar100c
import yaml

class CorruptionCIFAR(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        all_severity = [all_severity] if not isinstance(all_severity, list) else all_severity

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None
        if cfg.CORRUPTION.DATASET == "cifar10c":
            self.load_image = load_cifar10c
        elif cfg.CORRUPTION.DATASET == "cifar100c":
            self.load_image = load_cifar100c
        self.domain_id_to_name = {}
        data_source = []
        
        for i_s, severity in enumerate(self.severity):
            for i_c, corruption in enumerate(self.corruptions):
                d_name = f"{corruption}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name
                x, y = self.load_image(cfg.CORRUPTION.NUM_EX,
                                       severity,
                                       cfg.DATA_DIR,
                                       False,
                                       [corruption])
                for i in range(len(y)):
                    data_item = DatumRaw(x[i], y[i].item(), d_id)
                    data_source.append(data_item)

        super().__init__(cfg, data_source)

class CorruptionCIFARRecur(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        all_severity = [all_severity] if not isinstance(all_severity, list) else all_severity

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None
        dts = cfg.CORRUPTION.DATASET.split("_")[0]
        if dts == "cifar10c":
            self.load_image = load_cifar10c
        elif dts == "cifar100c":
            self.load_image = load_cifar100c
        self.domain_id_to_name = {}
        
        if cfg.CORRUPTION.ORDER_FILE is not None:
            with open(cfg.CORRUPTION.ORDER_FILE, 'r') as f:
                recur_corruptions = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            recur_corruptions = {i: self.corruptions for i in range(cfg.CORRUPTION.RECUR)}

        data_source = []
        for r in range(cfg.CORRUPTION.RECUR):
            for i_s, severity in enumerate(self.severity):
                for i_c, corruption in enumerate(recur_corruptions[r]):
                    d_name = f"{corruption}_{severity}_rep{r}"
                    d_id = r * (len(recur_corruptions[r]) * len(self.severity)) + i_s * len(recur_corruptions[r]) + i_c
                    self.domain_id_to_name[d_id] = d_name

                    x, y = self.load_image(cfg.CORRUPTION.NUM_EX,
                                        severity,
                                        cfg.TTA_DATA_DIR,
                                        False,
                                        [corruption])
                    
                    for i in range(len(y)):
                        data_item = DatumRaw(x[i], y[i].item(), d_id)
                        data_source.append(data_item)

        super().__init__(cfg, data_source)