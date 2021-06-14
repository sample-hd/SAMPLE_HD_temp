import setup

import argparse
import json

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', required=True)
parser.add_argument('--output_file', '-o', required=True)


def main(args):
    with open(args.input_file, 'r') as f:
        instr_file = json.load(f)
    # info = instr_file['info']
    instructions = instr_file['instructions']

    for ins in tqdm(instructions):
        if ins['template_overall_idx'] in [0, 4, 6]:
            targets = ["<OBJ1>"]
        elif ins['template_overall_idx'] in [1, 2, 3, 7, 8, 9]:
            targets = ["<OBJ1>", "<OBJ2>"]
        else:
            targets = []

        task_id = ins['task_id']

        if task_id in ['place', 'put', 'stack']:
            task_id = ins['task'][0]['type']
            if task_id == 'place':
                task_name = task_id + '[' + ins['task'][0]['subject'] + ']'
            else:
                task_name = task_id
            program = [task_name]
            for tar in targets:
                tar_prog = ins['program'][tar]
                program.append('scene')
                for p in tar_prog:
                    program.append(
                        p['type'] + '[' + p['input_value'] + ']'
                    )

        elif task_id in ['remove']:
            subject = ins['task'][0]['subject']
            if subject == 'left':
                subject_tar = 'right'
            elif subject == 'right':
                subject_tar = 'left'
            elif subject == 'front':
                subject_tar = 'back'
            elif subject == 'back':
                subject_tar = 'front'
            program = [
                'place' + '[' + subject_tar + ']',
                'scene',
                'filter_table[' + subject + ']'
            ]
        else:
            raise NotImplementedError()

        ins['program_sequence'] = program

    with open(args.output_file, 'w') as f:
        json.dump(instr_file, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
