import setup
import os
import json
import copy
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', required=True)
parser.add_argument('--start_idx', '-si', type=int, default=0)
parser.add_argument('--num_sequences', '-ns', type=int, default=-1)
parser.add_argument('--mid_frames', '-f', type=int, default=1)

parser.add_argument('--output_file', '-o', required=True)
parser.add_argument('--update_data', default=False, action='store_true')


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
        seq_path = os.path.join(seq_dir, seq_name, 'data_subtasks.json')
        effector_path = os.path.join(seq_dir, seq_name, 'effector.json')
        # print(seq_path)
        with open(seq_path, 'r') as f:
            data_f = json.load(f)
        with open(effector_path, 'r') as f:
            effector_data = json.load(f)['data']

        seq_data = data_f['sequence']
        subtask_data = data_f['subtasks']
        if args.update_data:
            new_data_full = {
                'scene_json': data_f['scene_json'],
                'sequence': [],
                'subtasks': subtask_data
            }
        new_data_redu = {
            'scene_json': data_f['scene_json'],
            'sequence': []
        }

        chosen_frames = []
        chosen_frames.append(subtask_data[0]["subtask_start"])
        for subt in subtask_data:
            end_frame = subt['subtask_end']
            mid_frames = []
            start_frame = subt['subtask_start']
            frame_len = end_frame - subt['subtask_start'] + 1
            for i in range(args.mid_frames + 1):
                frame_offset = int(frame_len * (i + 1) / (args.mid_frames + 1)) - 1
                frame_num = start_frame + frame_offset
                # print("{}\t{}\t{}\t{}\t{}".format(start_frame, end_frame, frame_len, frame_offset, frame_num))
                mid_frames.append(frame_num)
            chosen_frames += mid_frames
            # print(chosen_frames)

        for frame in seq_data:
            frame['effector_end'] = effector_data[frame['frame_number']]
            if args.update_data:
                new_data_full['sequence'].append(frame)
            if frame['frame_number'] in chosen_frames:
                new_data_redu['sequence'].append(frame)

        redu_seq_path = os.path.join(seq_dir, seq_name, args.output_file)
        # new_seq_path = os.path.join(seq_dir, seq_name, 'data_full.json')
        new_seq_path = seq_path

        if args.update_data:
            with open(new_seq_path, 'w') as f:
                json.dump(new_data_full, f, indent=4)

        with open(redu_seq_path, 'w') as f:
            json.dump(new_data_redu, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
