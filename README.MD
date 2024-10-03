# PeTTA: Persistent Test-time Adaptation in Recurring Testing Scenarios (NeurIPS 2024)

This repository is the official implementation of [Persistent Test-time Adaptation in Recurring Testing Scenarios](https://arxiv.org/abs/2311.18193) accepted at NeurIPS 2024. 

[[Project Page]](https://hthieu166.github.io/petta)

Authors: [Trung-Hieu Hoang](https://hthieu.web.illinois.edu/)<sup>1</sup>, [Duc Minh Vo](https://vmdlab.github.io/)<sup>2</sup>, [Minh N. Do](https://minhdo.ece.illinois.edu/)<sup>1</sup>

<sup>1</sup> University of Illinois at Urbana-Champaign,
<sup>2</sup> The University of Tokyo

## Requirements

To install requirements:

```setup
conda create -n petta python=3.9 pip
conda activate petta
pip install -r requirements.txt
```

## Recurring Test-time Adaptation (TTA)

To evaluate the test-time adaptation performance of a given adapter, run:
```syntax
./run.sh DATASET ADAPTER [CONFIG_FILE]
```
Available options for `DATASET` are: 

* [cifar10c](https://github.com/hendrycks/robustness?tab=readme-ov-file)
* [cifar100c](https://github.com/hendrycks/robustness?tab=readme-ov-file)
* [imagenetc](https://github.com/hendrycks/robustness?tab=readme-ov-file)
* [domainnet126](https://ai.bu.edu/M3SDA/)

*Note:* 

* After downloading the datasets, please modify the corresponding data path (e.g., `TTA_DATA_DIR` for testing dataset and `SRC_DATA_DIR` for source dataset) of dataset config files under `configs/dataset` folder.
* Pre-trained model on source dataset for DomainNet126 dataset can be downloaded from the official repository of [AdaContrast](https://github.com/DianCh/AdaContrast).  Structurally, it should be saved to the directory `./ckpt/domainnet126`.

Available options for `ADAPTER` are:

* Source model *(no TTA)*
* [RoTTA](https://github.com/BIT-DA/RoTTA)
* PeTTA *(ours)*

Please set `ADAPTER` to `petta`for running our PeTTA method. If `[CONFIG_FILE]` is not provided, default options will be used.
For example:
```eval
./run.sh cifar10c petta
./run.sh cifar100c rotta
./run.sh imagenetc source
```

## Results

We achieved the average classification error (lower is better) on the four *recurring* TTA (20 visits) tasks as follows:
<center>

| Method             | CF-10 $\rightarrow$ CF-10-C  | CF-100 $\rightarrow$ CF-100-C | DN126: R $\rightarrow$ C,P,S| IN $\rightarrow$ IN-C|
| ------------------ |:---------------------------: | :---------------------------: | :------------------: | :------------------: | 
| PeTTA *(ours)*     |               **22.8**       |      **35.1**                 |      **42.9**        |      **60.5**        |
| Source             |                43.5          |      46.5                     |        45.3          |      82.0            |
| RoTTA              |                51.3          |      61.4                     |        62.1          |      87.9            |

</center>

*Note:* The script `export_text_table.py` can parse the ouput evaluation files and generate the average performance within each repetitions, similar to the tables provided in the paper.

## Bibtex
If you use this code or found it helpful, please consider citing:
```
@inproceedings{hoang2024petta,
  title = {Persistent Test-time Adaptation in Recurring Testing Scenarios},
  author = {Trung-Hieu Hoang and Duc Minh Vo and Minh N. Do},
  booktitle = {Thirty-eighth Conference on Neural Information Processing Systems (NeurIPS)},
  year = {2024},
}
```

## Acknowledgements
The implementation of this repository is based on the following open-source projects:
* [RoTTA](https://github.com/BIT-DA/RoTTA)
* [Online Test-time Adaptation](https://github.com/mariodoebler/test-time-adaptation)