import os
import json
import argparse
import copy


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', required=True)
parser.add_argument('--instruction_json', '-ins', required=True)


def main(args):
    seq_dir_list = sorted(os.listdir(args.input_dir))
    # print(seq_dir_list)
    instr_nums = [
        int(seq_name[1:7]) for seq_name in seq_dir_list
    ]
    # print(instr_nums)
    with open(args.instruction_json, 'r') as f:
        instr_struct = json.load(f)
    new_struct = copy.deepcopy(instr_struct)
    new_struct['instructions'] = []
    for instr in instr_struct['instructions']:
        if instr['instruction_idx'] in instr_nums:
            new_struct['instructions'].append(instr)

    fname, _ = os.path.splitext(args.instruction_json)
    new_name = fname + '_filt.json'
    with open(new_name, 'w') as f:
        json.dump(new_struct, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
