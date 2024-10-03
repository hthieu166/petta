from torch.utils.data import DataLoader
from .ttasampler import build_sampler

from src.data.cifar_c import CorruptionCIFAR, CorruptionCIFARRecur
from src.data.domainnet import DomainNet126
from src.data.imagenet_c import ImageNetC
from ..utils.result_precess import AvgResultProcessor

def build_loader(cfg, ds_name, all_corruptions, all_severity):
    if ds_name == "cifar10c" or ds_name == "cifar100c":
        dataset_class = CorruptionCIFAR
    elif ds_name == "cifar10c_recur" or ds_name == "cifar100c_recur":
        dataset_class = CorruptionCIFARRecur
    elif ds_name == "domainnet126" or ds_name == "domainnet126_recur":
        dataset_class = DomainNet126
    elif ds_name == "imagenetc" or ds_name == "imagenetc_recur":
        dataset_class = ImageNetC
    else:
        raise NotImplementedError(f"Not Implement for dataset: {cfg.CORRUPTION.DATASET}")

    ds = dataset_class(cfg, all_corruptions, all_severity)
    sampler = build_sampler(cfg, ds.data_source)

    loader = DataLoader(ds, cfg.TEST.BATCH_SIZE, sampler=sampler, num_workers=cfg.LOADER.NUM_WORKS)

    result_processor = AvgResultProcessor(ds.domain_id_to_name)

    return loader, result_processor
