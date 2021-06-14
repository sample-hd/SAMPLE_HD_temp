import os
import argparse
import json

import h5py
import numpy as np

import setup

import utils.instructions as preprocess_utils
import utils.utils as utils

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input_instructions_json', required=True)
parser.add_argument('--input_seq_dir', required=True)
parser.add_argument('--data_filename', default='data_subtasks.json')

parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=False, action='store_true')
parser.add_argument('--encode_unk', default=False, action='store_true')
parser.add_argument('--decapitalise', default=False, action='store_true')
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--num_samples', '-n', type=int, default=None)
parser.add_argument('--start_idx', '-s', type=int, default=None)

parser.add_argument('--output_h5_file', required=True)
parser.add_argument('--output_vocab_json', default='')


def main(args):
    if (args.input_vocab_json == '') and (args.output_vocab_json == ''):
        print('Must give one of --input_vocab_json or --output_vocab_json')
        return

    print('Loading data')
    with open(args.input_instructions_json, 'r') as f:
        instructions = json.load(f)['instructions']

    if args.num_samples is not None:
        instructions = instructions[:args.num_samples]

    seq_names_list = []
    seq_obj_targets_list = []
    seq_obj_masks_list = []

    for instr in tqdm(instructions):
        instr_idx = instr['instruction_idx']
        scene_idx = instr['image_index']
        seq_dir_name = 'i{:06d}_s{:06d}'.format(instr_idx, scene_idx)
        seq_dir_path = os.path.join(args.input_seq_dir, seq_dir_name)
        data_path = os.path.join(seq_dir_path, args.data_filename)
        with open(data_path, 'r') as f:
            data_struct = json.load(f)['subtasks']
        # print(data_struct)
        seq_names = [sub['subtask_name'] for sub in data_struct]
        seq_obj_targets = [sub['subtask_object'] for sub in data_struct]
        seq_obj_masks = [sub['subtask_target_mask'] for sub in data_struct]
        seq_names_list.append(seq_names)
        seq_obj_targets_list.append(seq_obj_targets)
        seq_obj_masks_list.append(seq_obj_masks)
        # print(seq_names)
        # print(seq_obj_targets)
        # print(seq_obj_masks)

    # Either create the vocab or load it from disk
    if args.input_vocab_json == '' or args.expand_vocab:
        print('Building vocab')

        instruction_token_to_idx = preprocess_utils.build_vocab(
            (instr['instruction'] for instr in instructions),
            min_token_count=args.unk_threshold,
            punct_to_keep=[';', ','], punct_to_remove=['?', '.'],
            decapitalise=args.decapitalise
        )

        subtask_token_to_idx = preprocess_utils.build_vocab(
            seq_names_list,
            min_token_count=args.unk_threshold,
            punct_to_keep=[';', ','], punct_to_remove=['?', '.'],
            decapitalise=False
        )

        # print(subtask_token_to_idx)

        vocab = {
            'instruction_token_to_idx': instruction_token_to_idx,
            'subtask_token_to_idx': subtask_token_to_idx,
        }

    if args.input_vocab_json != '':
        print('Loading vocab')
        if args.expand_vocab:
            new_vocab = vocab
        with open(args.input_vocab_json, 'r') as f:
            vocab = json.load(f)
        if args.expand_vocab:
            num_new_words = 0
            for word in new_vocab['instruction_token_to_idx']:
                if word not in vocab['instruction_token_to_idx']:
                    print('Found new word %s' % word)
                    idx = len(vocab['instruction_token_to_idx'])
                    vocab['instruction_token_to_idx'][word] = idx
                    num_new_words += 1

            for word in new_vocab['subtask_token_to_idx']:
                if word not in vocab['subtask_token_to_idx']:
                    print('Found new word %s' % word)
                    idx = len(vocab['subtask_token_to_idx'])
                    vocab['subtask_token_to_idx'][word] = idx
                    num_new_words += 1

            print('Found %d new words' % num_new_words)

    if args.output_vocab_json != '':
        utils.mkdirs(os.path.dirname(args.output_vocab_json))
        with open(args.output_vocab_json, 'w') as f:
            json.dump(vocab, f)

    # Encode all questions and programs
    print('Encoding data')
    instructions_encoded = []
    subtasks_encoded = []
    # question_families = []
    orig_idxs = []
    array_idxs = []
    image_idxs = []
    subtask_obj_targets = []
    subtask_mask_sizes = []
    subtask_mask_strings = []
    for orig_idx, instr in enumerate(instructions):
        if orig_idx < args.start_idx:
            continue
        instruction = instr['instruction']

        array_idxs.append(orig_idx)
        orig_idxs.append(instr['instruction_idx'])
        image_idxs.append(instr['image_index'])
        instruction_tokens = preprocess_utils.tokenize(instruction,
                                                       punct_to_keep=[';', ','],
                                                       punct_to_remove=['?', '.'],
                                                       decapitalise=args.decapitalise)
        instruction_encoded = preprocess_utils.encode(instruction_tokens,
                                                      vocab['instruction_token_to_idx'],
                                                      allow_unk=args.encode_unk)
        instructions_encoded.append(instruction_encoded)

        subtask_tokens = preprocess_utils.tokenize(
            ' '.join(seq_names_list[orig_idx]))
        subtask_encoded = preprocess_utils.encode(subtask_tokens,
                                                  vocab['subtask_token_to_idx'],
                                                  allow_unk=args.encode_unk)
        subtasks_encoded.append(subtask_encoded)

        # print(subtask_tokens)
        # print(subtasks_encoded)

        # print(seq_obj_targets_list[orig_idx])
        obj_targets = seq_obj_targets_list[orig_idx]
        obj_targets = [None] + obj_targets
        obj_targets = [idx if idx is not None else -1 for idx in obj_targets]
        subtask_obj_targets.append(obj_targets)
        # print(subtask_obj_targets)
        masks = seq_obj_masks_list[orig_idx]
        masks = [None] + masks
        # print(masks)
        sizes = [mask['size'] if mask is not None else [-1, -1] for mask in masks]
        strs = [mask['counts'] if mask is not None else '' for mask in masks]
        # print(sizes)
        # print(strs)
        # print(subtask_tokens)
        # print(subtask_encoded)
        # print(obj_targets)
        # print(strs)
        subtask_mask_sizes.append(sizes)
        subtask_mask_strings.append(strs)

    # Pad encoded questions and programs
    max_instruction_length = max(len(x) for x in instructions_encoded)
    instruction_lengths = [len(x) for x in instructions_encoded]
    for ie in instructions_encoded:
        while len(ie) < max_instruction_length:
            ie.append(vocab['instruction_token_to_idx']['<NULL>'])

    max_subtask_len = max(len(x) for x in subtasks_encoded)
    subtask_lengths = [len(x) for x in subtasks_encoded]
    # print(subtask_lengths)
    for se in subtasks_encoded:
        while len(se) < max_subtask_len:
            se.append(vocab['subtask_token_to_idx']['<NULL>'])
    for se in subtask_obj_targets:
        while len(se) < max_subtask_len:
            se.append(-1)
    for se in subtask_mask_sizes:
        while len(se) < max_subtask_len:
            se.append([-1, -1])
    for se in subtask_mask_strings:
        while len(se) < max_subtask_len:
            se.append('')

    # print(subtasks_encoded)
    # print(subtask_obj_targets)
    # print(subtask_mask_sizes)
    # print(subtask_mask_strings)


    # Create h5 file
    print('Writing output')
    instructions_encoded = np.asarray(instructions_encoded, dtype=np.int32)
    print(instructions_encoded.shape)
    utils.mkdirs(os.path.dirname(args.output_h5_file))
    # print(np.asarray(subtask_mask_strings))
    with h5py.File(args.output_h5_file, 'w') as f:
        f.create_dataset('instructions', data=instructions_encoded)
        f.create_dataset('instruction_lengths', data=instruction_lengths)
        f.create_dataset('image_idxs', data=np.asarray(image_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))
        f.create_dataset('array_idxs', data=np.asarray(array_idxs))
        f.create_dataset('subtasks', data=np.asarray(subtasks_encoded))
        f.create_dataset('subtask_lengths', data=np.asarray(subtask_lengths))
        f.create_dataset('subtask_obj_targets', data=np.asarray(subtask_obj_targets))
        f.create_dataset('subtask_mask_sizes', data=np.asarray(subtask_mask_sizes))
        f.create_dataset('subtask_mask_counts', data=np.asarray(subtask_mask_strings, dtype='S'))

    with h5py.File(args.output_h5_file, 'r') as f:
        # f.create_dataset('instructions', data=instructions_encoded)
        print(f['instructions'][:])
        print(f['instruction_lengths'][:])
        print(f['image_idxs'][:])
        print(f['orig_idxs'][:])
        print(f['array_idxs'][:])
        print(f['subtasks'][:])
        print(f['subtask_lengths'][:])
        print(f['subtask_obj_targets'][:])
        print(f['subtask_mask_sizes'][:])
        print(f['subtask_mask_counts'][:])
        # f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
