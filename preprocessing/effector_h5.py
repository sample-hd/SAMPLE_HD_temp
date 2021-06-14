import os
import argparse
import json

import h5py
import numpy as np

import setup

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', required=True)
parser.add_argument('--json_name', '-j', required=True)
parser.add_argument('--start_idx', '-si', type=int, default=0)
parser.add_argument('--num_sequences', '-ns', type=int, default=-1)

parser.add_argument('--output_file', '-o', required=True)


def main(args):
    seq_dir = args.input_dir
    seq_names = sorted(os.listdir(seq_dir))
    start_idx = args.start_idx
    if start_idx >= len(seq_names):
        return
    num_sequences = args.num_sequences
    if num_sequences < 0 or num_sequences > len(seq_names) - start_idx:
        num_sequences = len(seq_names) - start_idx

    positions = []
    names = []
    for i in tqdm(range(start_idx, start_idx + num_sequences)):
        seq_name = seq_names[i]
        seq_path = os.path.join(seq_dir, seq_name, args.json_name)
        # print(seq_path)
        with open(seq_path, 'r') as f:
            seq_data = json.load(f)['sequence']
            for frame in seq_data:
                effector_pos = frame['effector_end']['position']
                effector_pos = list(effector_pos.values())
                img_name = "sequence_img_{:05d}.jpg".format(
                    frame['frame_number']
                )
                img_path = os.path.join(seq_name, img_name)
                names.append(img_path)
                positions.append(effector_pos)

    # print(names)
    # print(positions)

    names = np.asarray(names, dtype='S')
    positions = np.asarray(positions)

    print("Number of samples:\t" + str(len(positions)))
    # print(names)
    # print(positions)

    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset("image_paths", data=names)
        f.create_dataset("positions", data=positions)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
