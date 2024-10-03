import torch
import argparse
from src.configs import cfg
from src.model import build_model
from src.data import build_loader
from src.optim import build_optimizer
from src.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle
import numpy as np
import os.path as osp
import os
import torch.multiprocessing
import pandas as pd
from src.utils import set_random_seed

def recurring_test_time_adaptation(cfg):
    # Building model, optimizer and adapter:
    model = build_model(cfg)
    
    # Building optimizer
    optimizer = build_optimizer(cfg)
    
    # Initializing TTA adapter
    tta_adapter = build_adapter(cfg)
    tta_model = tta_adapter(cfg, model, optimizer)
    tta_model.cuda()
    
    # Building data loader
    loader, processor = build_loader(cfg, cfg.CORRUPTION.DATASET, cfg.CORRUPTION.TYPE, cfg.CORRUPTION.SEVERITY)
    
    # Save logs
    outputs_arr = []
    labels_arr = []
    
    # Main test-time Adaptation loop # (The main for-loop on line #2 of Alg. 1)
    tbar = tqdm(loader)
    for batch_id, data_package in enumerate(tbar):
        data, label, domain = data_package["image"], data_package['label'], data_package['domain']
        
        if len(label) == 1:
            continue  # ignore the final single point
        
        data, label = data.cuda(), label.cuda()
        output = tta_model(data, label=label)
        
        outputs_arr.append(output.detach().cpu().numpy())
        labels_arr.append(label.detach().cpu().numpy())
        
        predict = torch.argmax(output, dim=1)
        accurate = (predict == label)
        processor.process(accurate, domain)
        
        tbar.set_postfix(acc=processor.cumulative_acc())

    labels_arr = np.concatenate(labels_arr, axis=0)
    outputs_arr = np.concatenate(outputs_arr, axis=0)

    processor.calculate()
    _, prcss_eval_csv = processor.info()
    return prcss_eval_csv, tta_model

def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        '-acfg',
        '--adapter-config-file',
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str)
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)
    
    # Parsing arguments
    args = parser.parse_args()
    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')
    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    setproctitle(f"TTA:{cfg.CORRUPTION.DATASET:>8s}:{cfg.ADAPTER.NAME:<10s}")

    # For reproducibility
    torch.backends.cudnn.benchmark = True
    set_random_seed(cfg.SEED)

    # Running recurring TTA
    prcss_eval_csv, _ = recurring_test_time_adaptation(cfg) 
  
    # Saving evaluation results to files:
    if cfg.OUTPUT_DIR:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok =True)
    
    log_file_name = "%s_%s" % (
        osp.basename(args.dataset_config_file).split('.')[0], 
        osp.basename(args.adapter_config_file).split('.')[0])
    
    with open(osp.join(cfg.OUTPUT_DIR, "%s.csv" % log_file_name), "w") as fo:
        fo.write(prcss_eval_csv)

if __name__ == "__main__":
    main()