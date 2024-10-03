import torch
import torch.nn.functional as F
import math
from copy import deepcopy
from torch import nn

def compute_feat_mean(feats, pseudo_lbls):
    lbl_uniq = torch.unique(pseudo_lbls)
    lbl_group = [torch.where(pseudo_lbls==l)[0] for l in lbl_uniq]
    group_avgs = []
    for i, lbl_idcs in enumerate(lbl_group):
        group_avgs.append(feats[lbl_idcs].mean(axis=0).unsqueeze(0))
    return lbl_uniq, group_avgs 

class DivergenceScore(nn.Module):
    def __init__(self, src_prototype, src_prototype_cov):
        super().__init__()
        self.src_proto = src_prototype
        self.src_proto_cov = src_prototype_cov
        def GSSLoss(input, target, target_cov):
            return ((input-target).pow(2) / (target_cov + 1e-6)).mean()
        self.lss =  GSSLoss
    
    def forward(self, feats, pseudo_lbls):
        lbl_uniq, group_avgs = compute_feat_mean(feats, pseudo_lbls)
        return self.lss(
            torch.cat(group_avgs, dim=0),
            self.src_proto[lbl_uniq],
            self.src_proto_cov[lbl_uniq],
            )

class PrototypeMemory:
    def __init__(self, src_prototype, num_classes) -> None:
        self.src_proto = src_prototype.squeeze(1)
        self.mem_proto = deepcopy(self.src_proto) 
        self.num_classes = num_classes
        self.src_proto_l2 = torch.cdist(self.src_proto, self.src_proto, p=2)
    
    def update(self, feats, pseudo_lbls, nu=0.05):
        lbl_uniq = torch.unique(pseudo_lbls)
        lbl_group = [torch.where(pseudo_lbls==l)[0] for l in lbl_uniq]
        for i, lbl_idcs in enumerate(lbl_group):
            psd_lbl = lbl_uniq[i]
            batch_avg =  feats[lbl_idcs].mean(axis=0)
            self.mem_proto[psd_lbl] = (1-nu) * self.mem_proto[psd_lbl] + nu * batch_avg

    def get_mem_prototype(self):
        return self.mem_proto


class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0, true_label=None):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age
        self.true_label = true_label

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"
    
# Adapted from: https://github.com/BIT-DA/RoTTA/blob/main/core/utils/memory.py
class PeTTAMemory:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u

        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def get_accuracy(self):
        acc = 0.0
        for clss, dat in enumerate(self.data):
            for item in dat:
                acc += (item.true_label == clss).item()
        return acc / (self.get_occupancy() + 1e-6)

    def per_class_dist(self):
        per_class_occupied = [0] * self.num_class
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)

        return per_class_occupied

    def add_instance(self, instance):
        assert (len(instance) == 4)
        x, prediction, uncertainty, true_label = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0, true_label=true_label)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes(majority_classes, score)
        else:
            return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes, score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainty = item.uncertainty
                age = item.age
                score = self.heuristic_score(age=age, uncertainty=uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls

        if max_class is not None:
            if max_score > score_base:
                d = self.data[max_class].pop(max_index)
                del d
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)

        return classes

    def heuristic_score(self, age, uncertainty):
        return self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) + self.lambda_u * uncertainty / math.log(self.num_class)

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)

        tmp_age = [x / self.capacity for x in tmp_age]

        return tmp_data, tmp_age