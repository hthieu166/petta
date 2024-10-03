import torch
import torch.nn as nn
from .base_adapter import BaseAdapter

class Source(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(Source, self).__init__(cfg, model, optimizer)
        
    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, label):
        # batch data
        with torch.no_grad():
            model.eval()
            ema_out = self.model(batch_data)
        return ema_out

    def configure_model(self, model: nn.Module):
        return model