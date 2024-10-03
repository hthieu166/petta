from src.adapter.base_adapter import BaseAdapter
from src.adapter.rotta import RoTTA
from src.adapter.petta import PeTTA
from src.adapter.source import Source

def build_adapter(cfg) -> type(BaseAdapter):
    if cfg.ADAPTER.NAME == "rotta":
        return RoTTA
    if cfg.ADAPTER.NAME == "petta":
        return PeTTA 
    if cfg.ADAPTER.NAME == "source":
        return Source
    else:
        raise NotImplementedError("Implement your own adapter")

