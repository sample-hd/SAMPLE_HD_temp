from .SAMPLE import *
from torch.utils.data import DataLoader


def get_dataset(cfg, split):
    path = cfg.DATASET.PATH
    h5_instructions = cfg.DATASET.H5_INSTRUCTIONS
    vocab_json = cfg.DATASET.VOCAB_PATH
    img_dir = cfg.DATASET.IMG_DIR
    img_prefix = cfg.DATASET.IMG_PREFIX

    idx_list = None
    effector_h5 = None
    if split == 'train':
        if cfg.DATASET.TRAIN_IDX_LIST != '':
            idx_list = get_idx_list(cfg.DATASET.TRAIN_IDX_LIST)
        if cfg.DATASET.TRAIN_EFFECTOR_H5 != '':
            effector_h5 = cfg.DATASET.TRAIN_EFFECTOR_H5
        seq2seq_h5 = cfg.DATASET.H5_INSTRUCTIONS_TRAIN
        annotation_path = cfg.DATASET.ANNOTATIONS_TRAIN
    if split == 'test':
        if cfg.DATASET.TEST_IDX_LIST != '':
            idx_list = get_idx_list(cfg.DATASET.TEST_IDX_LIST)
        if cfg.DATASET.TEST_EFFECTOR_H5 != '':
            effector_h5 = cfg.DATASET.TEST_EFFECTOR_H5
        seq2seq_h5 = cfg.DATASET.H5_INSTRUCTIONS_TEST
        annotation_path = cfg.DATASET.ANNOTATIONS_TEST

    # print(h5_instructions)
    # print(dataset_length)
    if cfg.DATASET.DATASET == 'SAMPLE_HD_Instructions':
        dataset = SAMPLE_Instructions(
            path, h5_instructions, vocab_json,
            img_dir, img_prefix, idx_list)
    elif cfg.DATASET.DATASET == 'SAMPLE_HD_Sequence_Ref':
        dataset = SAMPLE_sequence_reference(
            path, vocab_json, idx_list)
    elif cfg.DATASET.DATASET == 'SAMPLE_HD_Effector':
        dataset = SAMPLE_effector_pos(
            path, effector_h5, idx_list)
    elif cfg.DATASET.DATASET == 'SAMPLE_HD_Seq2seq':
        dataset = SAMPLE_instruction_program(
            seq2seq_h5, vocab_json)
    elif cfg.DATASET.DATASET == 'SAMPLE_HD_Annotations':
        dataset = SAMPLE_attributes(
            annotation_path, path)
    else:
        raise NotImplementedError('No other datasets implemented yet')

    return dataset


def get_dataloader(cfg, split):
    dataset = get_dataset(cfg, split)
    shuffle = cfg.MODEL.SHUFFLE if split == 'train' else False
    batch = cfg.CONFIG.BATCH if split == 'train' else cfg.CONFIG.BATCH_VAL
    loader = DataLoader(dataset=dataset, batch_size=batch,
                        shuffle=shuffle, num_workers=cfg.CONFIG.NUM_WORKERS)
    print("Loaded {} dataset, split: {} number of samples: {}".format(
        cfg.DATASET.DATASET,
        split,
        len(dataset)
    ))

    return loader


def get_idx_list(path):
    with open(path, 'r') as f:
        idx_list = f.readlines()
    idx_list = [int(i) for i in idx_list]
    return idx_list
