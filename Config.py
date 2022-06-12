import os.path as osp
import yaml
from yacs.config import CfgNode as CN

from src.python import ReadJson

_C = CN()
# ========= Basic setting ============
_C.BASIC = CN()
_C.BASIC.RANDOM_SEED = 42       # random seed
_C.BASIC.DEVICE = [0]           # device
_C.BASIC.NUM_WORKER = 0         # numer of workers
_C.BASIC.IMG_ROOT  = 'Dataset'
_C.BASIC.JSON_ROOT = 'Json_Data' # use json file to control input data

# ========= Training setting =========
_C.TRAIN = CN()
_C.TRAIN.JSON_NAME = 'Fold0'    # json file name
_C.TRAIN.LAB_LOC   = 'Train_Labels'
_C.TRAIN.VAL_LOC   = 'Train_Images'
_C.TRAIN.IMG_LOC   = ['Train_Images', 'Train_Himg', 'Train_norm']
_C.TRAIN.IMG_PROB  = [0.75, 0.1, 0.15]
_C.TRAIN.SAVE_ROOT = 'checkpoint'
_C.TRAIN.WEIGHT_NUM = 10        # number of saved pth files

# ========= Testing setting (Self split dataset) ====================
_C.SELF_TEST = CN()
_C.SELF_TEST.IMG_LOC = 'Train_Images'
_C.SELF_TEST.LAB_LOC = 'Train_Labels'

# ========= Testing setting (Public and Private dataset) ============
_C.PUB_TEST = CN()
_C.PUB_TEST.PUB_IMG_LOC = 'Public_Image'
_C.PUB_TEST.PRI_IMG_LOC = 'Private_Image'

# ========= Augmentation setting =====
_C.AUG = CN()

_C.AUG.PREPROCESS = CN()
_C.AUG.PREPROCESS.MEAN = [0.485, 0.456, 0.406]
_C.AUG.PREPROCESS.STD = [0.229, 0.224, 0.225]

_C.AUG.CROP = CN()
_C.AUG.CROP.PROB_STAS = 0.8
_C.AUG.CROP.RAND_RANGE = [800, 800]

_C.AUG.RESIZE = CN()
_C.AUG.RESIZE.SIZE = [960, 1728]

_C.AUG.PAD = CN()
_C.AUG.PAD.SIZE = [960, 1728]

_C.AUG.MOSAIC = CN()
_C.AUG.MOSAIC.PROB = 0.0
_C.AUG.MOSAIC.OUT_SIZE = [960, 1728]

_C.AUG.FLIPLR = CN()
_C.AUG.FLIPLR.PROB = 0.5

_C.AUG.FLIPUD = CN()
_C.AUG.FLIPUD.PROB = 0.5

_C.AUG.ROT = CN()
_C.AUG.ROT.PROB = 0.0

_C.AUG.BRIGHT = CN()
_C.AUG.BRIGHT.PROB = 0.1
_C.AUG.BRIGHT.FACTOR = 0.1

_C.AUG.NOISE = CN()
_C.AUG.NOISE.PROB = 0.1
_C.AUG.NOISE.SIGMA = 0.01

_C.AUG.AUTOAUG = CN()
_C.AUG.AUTOAUG.PROB = 1.0
_C.AUG.AUTOAUG.OPT = 1

# ============== model =================
_C.MODEL = CN()
_C.MODEL.NAME = 'unet'

_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.NAME = 'efficientnet_b0'
_C.MODEL.ENCODER.DEPTH = 5
_C.MODEL.ENCODER.DROP_RATE = 0.2     # drop out rate

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.NORM = 'instance'   # ['batch', 'layer', 'instance', 'group']
_C.MODEL.DECODER.NORM_GROUPS = 4     # only used when NORM = 'group'

_C.MODEL.DECODER.ACTIVATION = 'silu' # ['leakyrelu', 'relu', 'silu', 'mish', 'sigmoid']
_C.MODEL.DECODER.UPSCALE = 'upconv'  # ['interpolation', 'upconv']

_C.MODEL.DECODER.DEEP_SUPERVISION_DEPTH = 3 # greater or equal than 1

# ============== Loss function =========
_C.LOSS = CN()
_C.LOSS.NAME = 'DFL'             # ['DL', 'GDL', 'DCEL', 'DFL']
_C.LOSS.ACT_KEY = 'softmax'      # act. function for the last layer and deep supervision layer
_C.LOSS.WEIGHT  = [1, 0.5, 0.25] # weight

# ======= optimizer and scheduler ======
_C.OPT = CN()

_C.OPT.SUB_BATCH_SIZE = 1
_C.OPT.UPDATE_FREQ = 8
_C.OPT.EPOCHS = 1

_C.OPT.OPTIMIZER = 'adamw'  # ['sgd', 'adam', 'adamw']
_C.OPT.LEARNING_RATE = 5e-4
_C.OPT.WEIGHT_DECAY = 1e-4

_C.OPT.WARMUP_EP = 5     # warm up epochs
_C.OPT.SCHEDULER = 'cos' # ['step', 'cos']

# The following two params are for StepLR scheduler
_C.OPT.DECAY_EP  = 15    # decay lr after per n epochs
_C.OPT.GAMMA     = 0.5   # decay multiplier

# The following two params are for CosineAnnealingLR scheduler
_C.OPT.ETA_MIN   = 1e-7   # minimum learning rate after T_max iterations was reached


# ============== Freeze ================
_C.freeze()

def get_cfg_defaults():
    Node = _C.clone()
    update_default_value(Node)
    return Node

def get_ckpt_cfg(ckpt_path, config_file='config.yaml'):
    Node = _C.clone()
    set_up_field(Node)
    Node.merge_from_file(osp.join(ckpt_path, config_file))
    update_default_value(Node)
    return Node

def set_up_field(Node):
    Node.defrost()
    Node.BASIC.TRAIN_DATA_NUM = None
    Node.AUG.MOSAIC.IN_SIZE = None
    Node.LOSS.LABEL_KEY = None
    Node.LOSS.PRED_KEY = None
    Node.LOSS.ACT_KEY = None
    Node.OPT.BATCH_SIZE = None
    Node.OPT.IT_PER_EP = None
    Node.OPT.WARMUP_IT = None
    Node.OPT.STEP_SIZE = None
    Node.OPT.T_MAX = None
    Node.freeze()
    return

def update_default_value(Node):
    Node.defrost()

    train_json = ReadJson(
        osp.join(Node.BASIC.JSON_ROOT, Node.TRAIN.JSON_NAME + '_train.json')
    )
    
    Node.BASIC.TRAIN_DATA_NUM = len(train_json)
    Node.AUG.MOSAIC.IN_SIZE = Node.AUG.RESIZE.SIZE
    Node.LOSS.PRED_KEY  = [f"stage_{i}" for i in range(Node.MODEL.DECODER.DEEP_SUPERVISION_DEPTH)]
    Node.LOSS.LABEL_KEY = [f"label_{i}" for i in range(Node.MODEL.DECODER.DEEP_SUPERVISION_DEPTH)]
    Node.LOSS.ACT_KEY   = [Node.LOSS.ACT_KEY] * Node.MODEL.DECODER.DEEP_SUPERVISION_DEPTH
    Node.OPT.BATCH_SIZE = Node.OPT.SUB_BATCH_SIZE * Node.OPT.UPDATE_FREQ
    Node.OPT.IT_PER_EP = Node.BASIC.TRAIN_DATA_NUM / Node.OPT.BATCH_SIZE
    Node.OPT.WARMUP_IT = Node.OPT.WARMUP_EP * Node.OPT.IT_PER_EP  # warmup iteraions
    Node.OPT.STEP_SIZE = Node.OPT.DECAY_EP  * Node.OPT.IT_PER_EP  # STEPLR: decay lr after per n iterations
    Node.OPT.T_MAX = (Node.OPT.EPOCHS - Node.OPT.WARMUP_EP) * Node.OPT.IT_PER_EP  # COSLR: maximum number of iteration

    Node.freeze()
    return
