import os
from utils import utils
from .supervisor import Supervisor
from .instruction_stage.base_seq import BaseSeqModel, BaseSeqModelTrainer
from .steering_stage.base_steer import BaseSteering, BaseSteerModel
from .rnn_blocks.rnn_instruction_encoder import RNN_instruction_encoder
from .rnn_blocks.rnn_instruction_decoder import RNN_instruction_decoder, RNN_instruction_decoder_attention
from .img_encoding.resnet_encoders import ResNetEncoder, ResNetEncoderNoLin
from .seq2seq.seq2seq import Seq2seqModel, Seq2seqTrainer
from .embeddings.subtask_mask import BasicEmbedding, BasicEmbeddingNoSeq
from .embeddings.decode import SimpleDecoder
from .effector_track.effector_track import EffectorTracking, EffectorTrackingModel
from .attributes.attribute_net import AttributeTrainer, AttributeModel
from .losses import *


def get_supervisor(cfg, model, train_loader, val_loader):
    logdir = cfg.LOGGING.LOGDIR
    logdir = utils.timestamp_dir(logdir)
    print("Logdir path: {}".format(logdir))
    with open(os.path.join(logdir, 'config.yaml'), 'w') as f:
        print(cfg, file=f)
    cfg_name = os.path.basename(cfg.INFO.CFG_PATH)
    utils.copy_file(cfg.INFO.CFG_PATH, os.path.join(logdir, cfg_name))
    utils.mkdirs(os.path.join(logdir, 'code'))
    # print(cfg.INFO.PROJECT_DIR)
    utils.copy_dirs(
        [os.path.join(cfg.INFO.PROJECT_DIR, dir_name) for dir_name in [
            'config',
            'datasets',
            'model',
            'preprocessing',
            'tools',
            'utils'
        ]],
        os.path.join(logdir, 'code')
    )
    if cfg.CONFIG.CHECKPOINT_PATH != '':
        checkpoint_path = cfg.CONFIG.CHECKPOINT_PATH
    else:
        checkpoint_path = None
    sv = Supervisor(train_loader, val_loader, model, logdir=logdir,
                    epochs=cfg.CONFIG.EPOCHS,
                    tensorboard=cfg.LOGGING.TENSORBOARD,
                    debug_log=cfg.LOGGING.DEBUG,
                    debug_freq=cfg.LOGGING.DEBUG_MSG,
                    display_freq=cfg.LOGGING.DISPLAY,
                    checkpoint_freq=cfg.CONFIG.CHECKPOINT,
                    load_path=checkpoint_path,
                    zero_counters=cfg.CONFIG.ZERO_TRAINING)

    return sv


def get_model(cfg):
    if cfg.MODEL.MODE == 'train':
        optimiser = {
            'name': cfg.MODEL.OPTIMISER,
            'learning_rate': cfg.MODEL.LEARNING_RATE,
            'weight_decay': cfg.MODEL.WEIGHT_DECAY
        }
    else:
        optimiser = None
    loss_name = cfg.MODEL.LOSS.NAME
    if loss_name == "TokenMaskLoss":
        factor = cfg.MODEL.LOSS.MASK_FACTOR
        token = cfg.MODEL.LOSS.TOKEN
        mask = cfg.MODEL.LOSS.MASK
        masked_mask = cfg.MODEL.LOSS.MASKED_MASK
        token_len_limit = cfg.MODEL.LOSS.TOKEN_LEN_LIMIT
        pos_w = cfg.MODEL.LOSS.POSITIVE_WEIGHT
        loss = TokenMaskLoss(factor, token, mask,
                             masked_mask, token_len_limit, pos_w)
    elif loss_name == "MSE_Multi":
        factor = cfg.MODEL.LOSS.PROGRESS_WEIGHT
        loss = MSE_Multi(factor)
    elif loss_name == "L2":
        loss = nn.MSELoss()
    elif loss_name == "L1":
        loss = nn.L1Loss()
    elif loss_name == 'CE':
        loss = nn.CrossEntropyLoss()
    elif loss_name == 'AnnL1':
        loss = AnnotationLoss("L1", cfg.MODEL.LOSS.POS_WEIGHT)
    elif loss_name == 'AnnL2':
        loss = AnnotationLoss("L2", cfg.MODEL.LOSS.POS_WEIGHT)

    if cfg.MODEL.ACTIVATION == '':
        activation = None
    elif cfg.MODEL.ACTIVATION == 'relu':
        activation = nn.ReLU(inplace=True)

    if cfg.MODEL.MODEL == 'PositionPred':
        img_encoder = ResNetEncoder(
            cfg.MODEL.IMG_ENCODER.OUT_SIZE,
            cfg.MODEL.IMG_ENCODER.NAME,
            cfg.MODEL.IMG_ENCODER.PRETRAINED,
            cfg.MODEL.IMG_ENCODER.DOUBLED_RES,
            cfg.MODEL.IMG_ENCODER.FREEZE,
        )

        model = EffectorTrackingModel(img_encoder)

        trainer = EffectorTracking(model, loss, optimiser)

    elif cfg.MODEL.MODEL == 'Annotation':
        img_encoder = ResNetEncoder(
            cfg.MODEL.IMG_ENCODER.OUT_SIZE,
            cfg.MODEL.IMG_ENCODER.NAME,
            cfg.MODEL.IMG_ENCODER.PRETRAINED,
            cfg.MODEL.IMG_ENCODER.DOUBLED_RES,
            cfg.MODEL.IMG_ENCODER.FREEZE,
        )

        model = AttributeModel(img_encoder)

        trainer = AttributeTrainer(model, loss, optimiser)

    elif cfg.MODEL.MODEL == 'Seq2seq':
        rnn_encoder = RNN_instruction_encoder(
            cfg.MODEL.INSTRUCTION_ENCODER.VOCAB_SIZE,
            cfg.MODEL.INSTRUCTION_ENCODER.WORD_EMBEDDING,
            cfg.MODEL.INSTRUCTION_ENCODER.HIDDEN_DIM,
            cfg.MODEL.INSTRUCTION_ENCODER.LAYERS,
            cfg.MODEL.INSTRUCTION_ENCODER.INPUT_DROPOUT,
            cfg.MODEL.INSTRUCTION_ENCODER.DROPOUT,
            rnn_cell=cfg.MODEL.INSTRUCTION_ENCODER.RNN_CELL
        )

        rnn_decoder = RNN_instruction_decoder_attention(
            cfg.MODEL.INSTRUCTION_DECODER.OUT_SIZE,
            cfg.MODEL.INSTRUCTION_DECODER.MAX_LEN,
            cfg.MODEL.INSTRUCTION_DECODER.INPUT_SIZE,
            cfg.MODEL.INSTRUCTION_DECODER.HIDDEN_DIM,
            cfg.MODEL.INSTRUCTION_DECODER.LAYERS,
            cfg.MODEL.INSTRUCTION_DECODER.INPUT_DROPOUT,
            cfg.MODEL.INSTRUCTION_DECODER.DROPOUT,
            rnn_cell=cfg.MODEL.INSTRUCTION_DECODER.RNN_CELL
        )

        model = Seq2seqModel(rnn_encoder, rnn_decoder)

        trainer = Seq2seqTrainer(model, loss, optimiser)

    else:
        print("No model found")
        exit()
    return trainer
