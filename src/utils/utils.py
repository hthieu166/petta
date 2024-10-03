import random
import numpy as np
import errno
import os
import os.path as osp
import warnings
import torch


def set_random_seed(seed):
    if seed > 0 :
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        print('No file found at "{}"'.format(fpath))
    return isfile


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)


class AverageMeterMultiTargets(object):
    def __init__(self, target_names):
        self.target_names = target_names
        self.reset()
    
    def reset(self):
        self.count = {}
        self.sum = {}

    def update(self,acc, dom,n=1):
        for d in torch.unique(dom).numpy():
            if d not in self.count:
                self.count[d] = 0.0
                self.sum[d] = 0.0
            arr = acc[dom == d]
            self.count[d] += len(arr)
            self.sum[d] += arr.sum()
    
    def average(self, key="index"):
        assert key in ["index", "name"]
        avg = {i if key == "index" else self.target_names[i]: (self.sum[i] * 1.0 /self.count[i]).item() for i in self.count}
        avg["avg"] = np.array([avg[k] for k in avg]).mean()
        return avg
    
    def __repr__(self):
        acc = self.average()
        s = ""
        for i in self.target_names:
            if i not in acc:
                continue
            s += "%20s: %.04f" % (
                self.target_names[i],
                acc[i]
            )
            s += "\n"
        s += "-" * 30 + "\n"
        s += f'>>> TARGET average:   %.04f'  % acc["avg"]
        return s
    
    def to_dict(self):
        return self.average(key="name")