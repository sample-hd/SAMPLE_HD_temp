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
parser.add_argument('--input_instructions_json', '-i', required=True)

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

    if args.start_idx is None:
        start_idx = 0
    else:
        start_idx = args.start_idx

    end_idx = len(instructions) - 1
    if args.num_samples is not None:
        end = start_idx + args.num_samples - 1
        if end <= end_idx:
            end_idx = end

    # Either create the vocab or load it from disk
    if args.input_vocab_json == '' or args.expand_vocab:
        print('Building vocab')

        instruction_token_to_idx = preprocess_utils.build_vocab(
            (instr['instruction'] for instr in instructions),
            min_token_count=args.unk_threshold,
            punct_to_keep=[';', ','], punct_to_remove=['?', '.'],
            decapitalise=args.decapitalise
        )

        programs = []
        for instr in instructions:
            programs += instr['program_sequence']
        program_token_to_idx = preprocess_utils.build_vocab(
            programs,
            min_token_count=args.unk_threshold,
            punct_to_keep=[';', ','], punct_to_remove=['?', '.'],
            decapitalise=False
        )

        # print(subtask_token_to_idx)

        vocab = {
            'instruction_token_to_idx': instruction_token_to_idx,
            'program_token_to_idx': program_token_to_idx,
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

            for word in new_vocab['program_token_to_idx']:
                if word not in vocab['program_token_to_idx']:
                    print('Found new word %s' % word)
                    idx = len(vocab['program_token_to_idx'])
                    vocab['program_token_to_idx'][word] = idx
                    num_new_words += 1

            print('Found %d new words' % num_new_words)

    if args.output_vocab_json != '':
        utils.mkdirs(os.path.dirname(args.output_vocab_json))
        with open(args.output_vocab_json, 'w') as f:
            json.dump(vocab, f)

    # Encode all questions and programs
    print('Encoding data')
    instructions_encoded = []
    programs_encoded = []
    # question_families = []
    orig_idxs = []
    array_idxs = []
    image_idxs = []
    for orig_idx, instr in enumerate(instructions):
        if orig_idx < start_idx:
            continue
        if orig_idx > end_idx:
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

        program_tokens = preprocess_utils.tokenize(' '.join(instr['program_sequence']),
                                                   punct_to_keep=[';', ','],
                                                   punct_to_remove=['?', '.'],
                                                   decapitalise=False)
        program_encoded = preprocess_utils.encode(program_tokens,
                                                  vocab['program_token_to_idx'],
                                                  allow_unk=args.encode_unk)
        programs_encoded.append(program_encoded)

    # Pad encoded questions and programs
    max_instruction_length = max(len(x) for x in instructions_encoded)
    instruction_lengths = [len(x) for x in instructions_encoded]
    for ie in instructions_encoded:
        while len(ie) < max_instruction_length:
            ie.append(vocab['instruction_token_to_idx']['<NULL>'])

    max_program_len = max(len(x) for x in programs_encoded)
    program_lengths = [len(x) for x in programs_encoded]
    # print(subtask_lengths)
    for pe in programs_encoded:
        while len(pe) < max_program_len:
            pe.append(vocab['program_token_to_idx']['<NULL>'])

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
        f.create_dataset('programs', data=np.asarray(programs_encoded))
        f.create_dataset('program_lengths', data=np.asarray(program_lengths))

    with h5py.File(args.output_h5_file, 'r') as f:
        # f.create_dataset('instructions', data=instructions_encoded)
        print(f['instructions'][0:5])
        print(f['instruction_lengths'][0:5])
        print(f['image_idxs'][0:5])
        print(f['orig_idxs'][0:5])
        print(f['array_idxs'][0:5])
        print(f['programs'][0:5])
        print(f['program_lengths'][0:5])
        print(f['orig_idxs'][0])
        print(f['array_idxs'][0])
        print(f['orig_idxs'][-1])
        print(f['array_idxs'][-1])
        # f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
