import os
from yacs.config import CfgNode as CN
from .paths_catalogue import DatasetsCatalog

_C = CN()

# Options for the choice of data
_C.DATASET = CN()

_C.DATASET.DATASET = 'SAMPLE_HD_Instructions'
# If you want to override dataset path set this option
# otherwise path will be set according to paths catalogue
_C.DATASET.PATH = ''
_C.DATASET.VOCAB_PATH = ''
_C.DATASET.H5_INSTRUCTIONS = ''
_C.DATASET.H5_INSTRUCTIONS_TRAIN = ''
_C.DATASET.H5_INSTRUCTIONS_TEST = ''
_C.DATASET.IMG_DIR = ''
_C.DATASET.TRAIN_IDX_LIST = ''
_C.DATASET.TRAIN_EFFECTOR_H5 = ''
_C.DATASET.TEST_IDX_LIST = ''
_C.DATASET.TEST_EFFECTOR_H5 = ''
_C.DATASET.IMG_PREFIX = 'SAMPLE_HD_train_'
_C.DATASET.ANNOTATIONS_TRAIN = ''
_C.DATASET.ANNOTATIONS_TEST = ''

# Model general options
_C.MODEL = CN()

_C.MODEL.MODEL = 'Stage1Base'
# Initial mode of the model
_C.MODEL.MODE = 'train'
# Dataloader shuffling (active only for train split)
_C.MODEL.SHUFFLE = True
# Optimiser
_C.MODEL.OPTIMISER = 'Adam'
# Learning rate
_C.MODEL.LEARNING_RATE = 0.001
_C.MODEL.WEIGHT_DECAY = 5e-4

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.VOCAB_SIZE = 81
_C.MODEL.INSTRUCTION_ENCODER.WORD_EMBEDDING = 128
_C.MODEL.INSTRUCTION_ENCODER.HIDDEN_DIM = 256
_C.MODEL.INSTRUCTION_ENCODER.LAYERS = 2
_C.MODEL.INSTRUCTION_ENCODER.INPUT_DROPOUT = 0.0
_C.MODEL.INSTRUCTION_ENCODER.DROPOUT = 0.0
_C.MODEL.INSTRUCTION_ENCODER.RNN_CELL = 'gru'

_C.MODEL.IMG_ENCODER = CN()
_C.MODEL.IMG_ENCODER.OUT_SIZE = 512
_C.MODEL.IMG_ENCODER.NAME = 'resnet50'
_C.MODEL.IMG_ENCODER.PRETRAINED = True
_C.MODEL.IMG_ENCODER.DOUBLED_RES = True
_C.MODEL.IMG_ENCODER.FREEZE = True

_C.MODEL.INSTRUCTION_DECODER = CN()
_C.MODEL.INSTRUCTION_DECODER.OUT_SIZE = 512
_C.MODEL.INSTRUCTION_DECODER.MAX_LEN = 20
_C.MODEL.INSTRUCTION_DECODER.INPUT_SIZE = 256
_C.MODEL.INSTRUCTION_DECODER.HIDDEN_DIM = 512
_C.MODEL.INSTRUCTION_DECODER.LAYERS = 2
_C.MODEL.INSTRUCTION_DECODER.INPUT_DROPOUT = 0.0
_C.MODEL.INSTRUCTION_DECODER.DROPOUT = 0.0
_C.MODEL.INSTRUCTION_DECODER.RNN_CELL = 'gru'
_C.MODEL.INSTRUCTION_DECODER.SEPARATE_REMAP = False

_C.MODEL.OUTPUT_EMBEDDING = CN()
_C.MODEL.OUTPUT_EMBEDDING.VOCAB_SIZE = 12
_C.MODEL.OUTPUT_EMBEDDING.SUBTASK_EMB = 64
_C.MODEL.OUTPUT_EMBEDDING.IMG_RESIZE = (180, 320)
_C.MODEL.OUTPUT_EMBEDDING.MASK_EMB = 256
_C.MODEL.OUTPUT_EMBEDDING.OUT_SIZE = 256

_C.MODEL.OUTPUT_DECODE = CN()
_C.MODEL.OUTPUT_DECODE.EMB_SIZE = 512
_C.MODEL.OUTPUT_DECODE.VOCAB_SIZE = 12
_C.MODEL.OUTPUT_DECODE.IMG_RESIZE = (180, 320)

_C.MODEL.ACTIVATION = 'relu'
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.NAME = 'TokenMaskLoss'
_C.MODEL.LOSS.MASK_FACTOR = 0.5
_C.MODEL.LOSS.TOKEN = 'NLL'
_C.MODEL.LOSS.MASK = 'BCE'
_C.MODEL.LOSS.MASKED_MASK = False
_C.MODEL.LOSS.TOKEN_LEN_LIMIT = False
_C.MODEL.LOSS.POSITIVE_WEIGHT = 1.0
_C.MODEL.LOSS.PROGRESS_WEIGHT = 0.7
_C.MODEL.LOSS.POS_WEIGHT = 0.7

_C.MODEL.TEST = CN()
_C.MODEL.TEST.MAX_LEN = 51
_C.MODEL.TEST.MASK_THRESHOLD = 0.7

# Logging
_C.LOGGING = CN()
# Logging dir - directory used for logging (will be appended with timestamp)
_C.LOGGING.LOGDIR = '../outputs/_logtest'
# Use of tensorboard (will be placed in logdir)
_C.LOGGING.TENSORBOARD = True
# Debug log presence
_C.LOGGING.DEBUG = True
# Debug message frequency (in iters)
_C.LOGGING.DEBUG_MSG = 10
# Display message frequency (in iters)
_C.LOGGING.DISPLAY = 20


# Other config options
_C.CONFIG = CN()
# Number of workers for loader
_C.CONFIG.NUM_WORKERS = 8
# Batch size
_C.CONFIG.BATCH = 256
_C.CONFIG.BATCH_VAL = 256
# Number of epochs
_C.CONFIG.EPOCHS = 100
# Checkpoint frequency (in number of iterations - may be better than epochs
# cause of better control) - or it probably does not matter tbh
_C.CONFIG.CHECKPOINT = 500
# Zeroing epoch and iteration counters, e.g. next stage of training
_C.CONFIG.ZERO_TRAINING = False
# Load path
_C.CONFIG.CHECKPOINT_PATH = ''
# Use of tensorboard
_C.CONFIG.TENSORBOARD = True
# Test split
_C.CONFIG.TEST_SPLIT = 'test'


# Some opts I used for debugging
_C.DEBUG = CN()
_C.DEBUG.TRAIN_SPLIT = 'train'
_C.DEBUG.TRAIN_LEN = 0
_C.DEBUG.VAL_SPLIT = 'val'
_C.DEBUG.VAL_LEN = 0


# Info options
_C.INFO = CN()
# Empty field to keep track of config file
_C.INFO.CFG_PATH = ''
# Empty field to keep track of project dir (usually ..)
_C.INFO.PROJECT_DIR = '..'
# Comment
_C.INFO.COMMENT = 'Default comment'


def get_cfg(config_file):
    # We don't touch _C as we may want to check defaults for w/e reason
    cfg = _C.clone()
    if config_file.lower() in ['default', 'defaults']:
        print('Returning default configuration')
    else:
        if os.path.isfile(config_file):
            cfg.merge_from_file(config_file)
        else:
            print('Incorrect path provided, loading defaults')

    cfg = fill_catalogue_paths(cfg)
    cfg.merge_from_list(['INFO.CFG_PATH', os.path.abspath(config_file)])
    # print(config_file, os.path.abspath(config_file))
    # if project_dir is not None:
        # print(project_dir, os.path.abspath(project_dir))
        # cfg.merge_from_list(['INFO.PROJECT_DIR', os.path.abspath(project_dir)])

    return cfg


def fill_catalogue_paths(cfg):
    if cfg.DATASET.PATH == '':
        paths = DatasetsCatalog.get(cfg.DATASET.DATASET)
        path, vocab_path, instruction_h5, instruction_raw, img_dir = paths
        cfg.merge_from_list(["DATASET.PATH", path])
    if cfg.DATASET.VOCAB_PATH == '':
        cfg.merge_from_list(["DATASET.VOCAB_PATH", vocab_path])
    if cfg.DATASET.H5_INSTRUCTIONS == '':
        cfg.merge_from_list(["DATASET.H5_INSTRUCTIONS", instruction_h5])
    if cfg.DATASET.IMG_DIR == '':
        cfg.merge_from_list(["DATASET.IMG_DIR", img_dir])

    return cfg
