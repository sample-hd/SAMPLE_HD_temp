import setup
import os
import json
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', required=True)
parser.add_argument('--data_name', '-fn', default='data_new.json')
parser.add_argument('--start_idx', '-si', type=int, default=0)
parser.add_argument('--num_sequences', '-ns', type=int, default=-1)


def main(args):
    seq_dir = args.input_dir
    seq_names = sorted(os.listdir(seq_dir))
    start_idx = args.start_idx
    if start_idx >= len(seq_names):
        return
    num_sequences = args.num_sequences
    if num_sequences < 0 or num_sequences > len(seq_names) - start_idx:
        num_sequences = len(seq_names) - start_idx

    for i in tqdm(range(start_idx, start_idx + num_sequences)):
        seq_name = seq_names[i]
        seq_path = os.path.join(seq_dir, seq_name, args.data_name)
        # print(seq_path)
        with open(seq_path, 'r') as f:
            data_f = json.load(f)

        seq_data = data_f['sequence']
        # print(subtask_list)
        subtask_path = os.path.join(seq_dir, seq_name, 'data_subtasks.json')

        for frame_info in seq_data:
            if "subtasks" not in data_f:
                data_f['subtasks'] = [frame_info]
            else:
                if 'subtask_name' in frame_info:
                    if frame_info['subtask_name'] != data_f['subtasks'][-1]['subtask_name']:
                        data_f['subtasks'].append(frame_info)

            # print(subtask_list_target)

        with open(subtask_path, 'w') as f:
            json.dump(data_f, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
