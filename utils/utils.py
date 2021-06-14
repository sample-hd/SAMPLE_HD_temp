import os
import json
import shutil
import numpy as np
from datetime import datetime


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def copy_file(src, dst):
    shutil.copyfile(src, dst)


def copy_dirs(src_list, dst):
    for d in src_list:
        shutil.copytree(d, os.path.join(dst, os.path.basename(d)), symlinks=True)


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['instruction_idx_to_token'] = invert_dict(vocab['instruction_token_to_idx'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['instruction_token_to_idx']['<NULL>'] == 0
    assert vocab['instruction_token_to_idx']['<START>'] == 1
    assert vocab['instruction_token_to_idx']['<END>'] == 2
    return vocab


def timestamp_dir(logdir):
    main_dir, exp_dir = os.path.split(logdir)
    # Append 'timestamp' to the experiment directory name
    now = datetime.now()
    yy = now.year % 100
    m = now.month
    dd = now.day
    hh = now.hour
    mm = now.minute
    ss = now.second
    timestamp = "{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(dd, m, yy, hh, mm, ss)
    exp_dir = "{}_{}".format(exp_dir, timestamp)
    logdir = os.path.join(main_dir, exp_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir
