"""
Adapted from: https://github.com/BIT-DA/RoTTA/blob/main/core/data/ttasampler.py
"""
import numpy as np
from torch.utils.data.sampler import Sampler
from src.data.base_dataset import DatumBase
from typing import List
from collections import defaultdict
from numpy.random import dirichlet

class LabelDirichletDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase], gamma, batch_size, slots=None):

        self.domain_dict = defaultdict(list)
        self.classes = set()
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
            self.classes.add(item.label)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_class = len(self.classes)
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []
        for domain in self.domains:
            indices = np.array(self.domain_dict[domain])
            labels = np.array([self.data_source[i].label for i in indices])

            class_indices = [np.argwhere(labels == y).flatten() for y in range(self.num_class)]
            slot_indices = [[] for _ in range(self.num_slots)]

            label_distribution = dirichlet([self.gamma] * self.num_slots, self.num_class)

            for c_ids, partition in zip(class_indices, label_distribution):
                for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int))):
                    slot_indices[s].append(ids)

            for s_ids in slot_indices:
                permutation = np.random.permutation(range(len(s_ids)))
                ids = []
                for i in permutation:
                    ids.extend(s_ids[i])
                final_indices.extend(indices[ids])
        return iter(final_indices)


class DirichletEpisodicDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase], gamma, batch_size, slots=5):

        self.domain_dict = defaultdict(list)
        self.label_dict = defaultdict(list)
        self.classes = set()
        
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)

        # Randomly shuffling items in each domain group
        for v in self.domain_dict:
            np.random.shuffle(self.domain_dict[v])

        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_class = len(self.classes)
        self.num_episode = 5
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []    
        domain_idcs = [self.domain_dict[y] for y in self.domains]
        slot_indices = [[] for _ in range(self.num_slots)]
        domain_distribution = np.random.dirichlet([self.gamma] * self.num_slots, len(self.domains))
        
        for c_ids, partition in zip(domain_idcs, domain_distribution):
            for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int))):
                slot_indices[s].append(ids)
        final_indices = []
        for s_ids in slot_indices:
            permutation = np.random.permutation(range(len(s_ids)))
            ids = []
            for i in permutation:
                ids.extend(s_ids[i])
            final_indices.extend(ids)
        
        return iter(final_indices)

class TemporalDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase]):

        self.domain_dict = defaultdict(list)
        self.label_dict = defaultdict(list)
        self.classes = set()
        
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)

        # Randomly shuffling items in each domain group
        for v in self.domain_dict:
            np.random.shuffle(self.domain_dict[v])

        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source
        self.num_class = len(self.classes)
        
    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []
        for y in self.domains:
            final_indices.extend(self.domain_dict[y])        
        return iter(final_indices)
    
def build_sampler(
        cfg,
        data_source: List[DatumBase],
        **kwargs
    ):
    if cfg.LOADER.SAMPLER.TYPE == "class_temporal":
        return LabelDirichletDomainSequence(data_source, cfg.LOADER.SAMPLER.GAMMA, cfg.TEST.BATCH_SIZE, **kwargs)
    else:
        raise NotImplementedError()
