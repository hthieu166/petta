from copy import deepcopy
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
from src.model.resnet_domainnet126 import ResNetDomainNet126

class BaseAdapter(nn.Module):
    def __init__(self, cfg, model, optimizer):
        super().__init__()
        self.cfg = cfg
        self.model = self.configure_model(model)

        params, param_names = self.collect_params(self.model)
        if len(param_names) == 0:
            self.optimizer = None
        else:
            self.optimizer = optimizer(params)

        self.steps = self.cfg.OPTIM.STEPS
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"

    def forward(self, x, label=None):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer, label)

        return outputs

    def forward_and_adapt(self, *args):
        raise NotImplementedError("implement forward_and_adapt by yourself!")

    def configure_model(self, model):
        raise NotImplementedError("implement configure_model by yourself!")

    def collect_params(self, model: nn.Module):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names

    def check_model(self, model):
        pass

    def before_tta(self, *args, **kwargs):
        pass

    def save_log(self, out_fold, file_name):
        pass
        
    @staticmethod
    def build_ema(model):
        is_parallel =  isinstance(model, nn.DataParallel)
        
        if is_parallel:
            model = model.module
    
        if isinstance(model, ResNetDomainNet126):  # https://github.com/pytorch/pytorch/issues/28594
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)
            ema_model = deepcopy(model)
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
        else:
            ema_model = deepcopy(model)
        
        for param in ema_model.parameters():
            param.detach_()
        
        if is_parallel:
            model = nn.DataParallel(model)
            ema_model = nn.DataParallel(ema_model)
            
        return ema_model