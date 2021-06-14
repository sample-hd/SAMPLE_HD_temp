import setup

import os
import json

from model.executor import Executor
from tqdm import tqdm


def check_same(ui_gt, ui_test):
    if ui_gt['image_filename'] != ui_test['image_filename']:
        return False
    if ui_gt['image_index'] != ui_test['image_index']:
        return False
    if ui_gt['instruction_idx'] != ui_test['instruction_idx']:
        return False
    if ui_gt['unity_program']['name'] != ui_test['unity_program']['name']:
        return False
    if ui_gt['unity_program']['subjectInt'] != ui_test['unity_program']['subjectInt']:
        return False
    if sorted(ui_gt['unity_program']['objectInt']) != sorted(ui_test['unity_program']['objectInt']):
        return False
    if sorted(ui_gt['unity_program']['subjectString']) != sorted(ui_test['unity_program']['subjectString']):
        return False
    return True


ex = Executor()

si = 0
ei = 9
si = 7912
ei = 10000

with open('/home/michas/Desktop/SAMPLE_HD/instructions/instructions_program.json', 'r') as f:
    instrs = json.load(f)['instructions']

with open('/home/michas/Desktop/SAMPLE_HD/instructions/instructions_unityGT_filtered.json', 'r') as f:
    instrs_unity = json.load(f)['instructions']

scene_dir = '/home/michas/Desktop/SAMPLE_HD/scenes'
# scene_dir = '/home/michas/Desktop/codes/nips2021/outputs/scenes'

correct = []
incorrect = []

for test_idx in tqdm(range(len(instrs))):
    # test_idx = 4
    if test_idx < si:
        continue
    if test_idx > ei:
        continue
    ins = instrs[test_idx]
    ins_u = instrs_unity[test_idx]
    true_idx = ins["instruction_idx"]
    # print(test_idx, true_idx)

    scene_name = os.path.join(scene_dir, "SAMPLE_HD_train_{:06d}.json".format(ins['image_index']))

    with open(scene_name, 'r') as f:
        scene_struct = json.load(f)
    ui = ex.parse_unity(ins['program_sequence'], scene_struct, true_idx)
    if not check_same(ui, ins_u):
        # print("DFSFDS")
        print(test_idx, true_idx)
        print(ui)
        print(ins_u)
        incorrect.append(test_idx)
        # break
    else:
        correct.append(test_idx)


print(correct)
print(incorrect)
