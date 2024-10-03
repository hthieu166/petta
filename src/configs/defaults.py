from yacs.config import CfgNode as CN
_C = CN()
cfg = _C
# ----------------------------- Model options ------------------------------- #
_C.MODEL = CN()
_C.MODEL.ARCH = 'Standard'
_C.MODEL.EPISODIC = False
_C.MODEL.PROJECTION = CN()
_C.MODEL.PROJECTION.HEAD = "linear"
_C.MODEL.PROJECTION.EMB_DIM = 2048
_C.MODEL.PROJECTION.FEA_DIM = 128
# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CN()
_C.CORRUPTION.DATASET = 'cifar10'
_C.CORRUPTION.SOURCE = ''
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]
_C.CORRUPTION.NUM_EX = 10000
_C.CORRUPTION.NUM_CLASS = -1
_C.CORRUPTION.RECUR = 10
_C.CORRUPTION.ORDER_FILE = None
# ----------------------------- Input options -------------------------- #
_C.INPUT = CN()
_C.INPUT.SIZE = (32, 32)
_C.INPUT.INTERPOLATION = "bilinear"
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.TRANSFORMS = ("normalize", )
# ----------------------------- loader options -------------------------- #
_C.LOADER = CN()
_C.LOADER.SAMPLER = CN()
_C.LOADER.SAMPLER.TYPE = "sequence"
_C.LOADER.SAMPLER.GAMMA = 0.1
_C.LOADER.SAMPLER.N_EPISODE = 10
_C.LOADER.NUM_WORKS = 2
# ------------------------------- Batch norm options ------------------------ #
_C.BN = CN()
_C.BN.EPS = 1e-5
_C.BN.MOM = 0.1
# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CN()
_C.OPTIM.STEPS = 1
_C.OPTIM.LR = 1e-3
_C.OPTIM.METHOD = 'Adam'
_C.OPTIM.BETA = 0.9
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.DAMPENING = 0.0
_C.OPTIM.NESTEROV = True
_C.OPTIM.WD = 0.0
# ------------------------------- Testing options --------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
# ---------------------------------- Misc options --------------------------- #
_C.DESC = ""
_C.SEED = 2020
_C.OUTPUT_DIR = "./output"
_C.TTA_DATA_DIR = "./datasets"
_C.SRC_DATA_DIR = "./datasets"
_C.CKPT_DIR = "./ckpt"
_C.LOG_DEST = "log.txt"
_C.CKPT_PATH = "./ckpt/domainnet126/best_real_2020.pth"
_C.LOG_TIME = ''
_C.DEBUG = 0

# tta method specific
_C.ADAPTER = CN()
_C.ADAPTER.NAME = "petta"

# RoTTA
_C.ADAPTER.RoTTA = CN()
_C.ADAPTER.RoTTA.MEMORY_SIZE = 64
_C.ADAPTER.RoTTA.UPDATE_FREQUENCY = 64
_C.ADAPTER.RoTTA.NU = 0.001
_C.ADAPTER.RoTTA.ALPHA = 0.05
_C.ADAPTER.RoTTA.LAMBDA_T = 1.0
_C.ADAPTER.RoTTA.LAMBDA_U = 1.0

# PETTA (ours) Configs 
_C.ADAPTER.PETTA = CN()
_C.ADAPTER.PETTA.ALPHA_0 = 0.001
_C.ADAPTER.PETTA.LAMBDA_0 = 0.0
_C.ADAPTER.PETTA.AL_WGT = 1.0
_C.ADAPTER.PETTA.REGULARIZER = "cosine"

_C.ADAPTER.PETTA.ADAPTIVE_LAMBDA = False
_C.ADAPTER.PETTA.ADAPTIVE_ALPHA = False

_C.ADAPTER.PETTA.NORM_LAYER = "rbn"
_C.ADAPTER.PETTA.LOSS_FUNC = "sce"

# Percentage of source samples used
_C.ADAPTER.PETTA.PERCENTAGE = 1.0   # [0, 1] possibility to reduce the number of source samples
_C.ADAPTER.PETTA.NUM_WORKERS = 4

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()