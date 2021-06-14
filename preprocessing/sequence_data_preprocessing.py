import setup
import os
import json
import copy
import argparse
import numpy as np
from tqdm import tqdm
from utils import robot
from PIL import Image
from pycocotools import coco

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', required=True)
parser.add_argument('--start_idx', '-si', type=int, default=0)
parser.add_argument('--num_sequences', '-ns', type=int, default=-1)
parser.add_argument('--scenes_dir', '-s', required=True)
parser.add_argument('--scenes_naming', '-sn', default="SAMPLE_HD_train_")
parser.add_argument('--instructions', '-ins', required=True)
parser.add_argument('--use_inter', '-inter', default=False, action='store_true')


def main(args):
    # tasks = []
    with open(args.instructions, 'r') as f:
        instructions = json.load(f)['instructions']

    # for ins in instructions:
    #     if len(ins['task']) not in tasks:
    #         tasks.append(len(ins['task']))

    # print(instructions)
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
        scene_num = int(seq_name[-6:])
        instr_num = int(seq_name[1:7])
        scene_name = args.scenes_naming + "{:06d}".format(scene_num) + ".json"
        scene_name = os.path.join(args.scenes_dir, scene_name)
        with open(scene_name, 'r') as f:
            scene_struct = json.load(f)
        # print(scene_struct)

        instr = [ins for ins in instructions if ins['instruction_idx'] == instr_num]
        instr = instr[0]
        task = instr['task'][0]
        # print(task)
        seq_path = os.path.join(seq_dir, seq_name, 'data.json')
        # print(seq_path)
        with open(seq_path, 'r') as f:
            data_f = json.load(f)

        seq_data = data_f['sequence']

        task_name = task['type']
        task_obj = task['object']
        task_subj = task['subject']

        # print(task_name, task_obj, task_subj)

        if task_name == 'remove':
            task_name = 'place'
            idx_list = []
            for idx, obj in enumerate(scene_struct['objects']):
                if task_subj in obj['table_part']:
                    idx_list.append(idx)
            if task_subj == 'left':
                tp = 'right'
            if task_subj == 'right':
                tp = 'left'
            if task_subj == 'front':
                tp = 'back'
            if task_subj == 'back':
                tp = 'front'
            task_obj = idx_list
            task_subj = tp

        # print(task_name, task_obj, task_subj)

        subtask_list = []
        for idx, data in enumerate(seq_data):
            subtask = data["subtask"]
            if len(subtask_list) < 1:
                subtask_list.append((subtask, idx))
            elif subtask_list[-1][0] != subtask:
                subtask_list[-1] = (subtask_list[-1][0], subtask_list[-1][1], idx - 1)
                subtask_list.append((subtask, idx))
            if idx == len(seq_data) - 1:
                subtask_list[-1] = (subtask_list[-1][0], subtask_list[-1][1], idx - 1)

        # print(subtask_list)

        subtask_list_target = []

        if len(task_obj) > 1:
            new_order = []
            approach_frames = [
                subt[1] for subt in subtask_list if subt[0] == "GripperMoveAvoid"
            ]
            effector_positions = [
                robot.UR5e.effector_end(seq_data[idx]['robot_state'][0:6]) for idx in approach_frames
            ]
            effector_positions = [
                np.array(pos) for pos in effector_positions
            ]
            obj_positions = [
                seq_data[0]['objects'][idx]["position"] for idx in task_obj
            ]
            # print(obj_positions)
            obj_positions = [
                np.array(list(pos.values())) for pos in obj_positions
            ]
            # print(obj_positions)
            # print(approach_frames)
            # print(effector_positions)
            while len(effector_positions) > 0:
                effector_end = effector_positions.pop(0)
                dists = [
                    np.linalg.norm(effector_end - pos) for pos in obj_positions
                ]
                # print(dists)
                min_idx = np.argmin(dists)
                # print(min_idx)
                new_order.append(task_obj[min_idx])
                # print(new_order)
                obj_positions.pop(min_idx)
                task_obj.pop(min_idx)
            task_obj = new_order

        # for idx in task_obj:
        #     print(scene_struct['objects'][idx]['name'])
        if args.use_inter:
            segmentation_dir = os.path.join(seq_dir, seq_name, 'segmentation_inter')
        else:
            segmentation_dir = os.path.join(seq_dir, seq_name, 'segmentation')
        # seg0 = os.path.join(segmentation_dir, 'sequence_img_00000.jpg')
        # img = Image.open(seg0)
        # # img.show()
        # data = np.asarray(img)
        # print(data.shape)
        # cols = []
        # for ii in range(data.shape[0]):
        #     for ij in range(data.shape[1]):
        #         col = tuple(data[ii, ij, :])
        #         if col not in cols:
        #             cols.append(col)

        # print(cols)
        # print(len(cols))

        mask_idxs = []

        for subt_idx, subt in enumerate(subtask_list):
            sub_name = subt[0]
            sub_start = subt[1]
            sub_end = subt[2]
            if sub_name == 'GripperMoveAvoid':
                seg_name = 'sequence_img_{:05d}.png'.format(sub_start)
                # print(seg_name)
                seg_map = np.asarray(Image.open(os.path.join(segmentation_dir, seg_name)))
                tar_col = scene_struct['objects'][task_obj[0]]['segmentation_colour']
                tar_col = list(tar_col.values())[0:-1]
                tar_col = np.asarray(tar_col)
                tar_col = np.expand_dims(tar_col, axis=0)
                tar_col = np.expand_dims(tar_col, axis=0)
                # print(seg_map.shape)
                # print(tar_col.shape)
                diff = seg_map - tar_col
                diff = np.linalg.norm(diff, axis=2)
                diff = np.uint8(diff < 3)
                mask = coco.maskUtils.encode(np.asfortranarray(diff))
                mask['counts'] = str(mask['counts'], "utf-8")
                # print(mask)
                # print(np.sum(diff))
                # im = Image.fromarray(np.uint8(diff)*255)
                # im.show()
                subtask_list_target.append(
                    [
                        sub_name,
                        sub_start,
                        sub_end,
                        task_obj[0],
                        mask
                    ]
                )
                mask_idxs.append(sub_start)
            elif sub_name == 'ApproachGrasp':
                # Get same mask as previous subtask
                subtask_list_target.append(
                    [
                        sub_name,
                        sub_start,
                        sub_end,
                        task_obj[0],
                        subtask_list_target[-1][-1]
                    ]
                )
            elif sub_name == 'CloseGripper':
                # No target
                subtask_list_target.append(
                    [
                        sub_name,
                        sub_start,
                        sub_end,
                        None,
                        None
                    ]
                )
            elif sub_name == 'MoveUP':
                # No target - maybe let it stay
                subtask_list_target.append(
                    [
                        sub_name,
                        sub_start,
                        sub_end,
                        None,
                        None
                    ]
                )
            elif sub_name == 'ObjectMove':
                frame_idx = subtask_list[-1][-1]
                # print("obj move", frame_idx)
                for j in range(subt_idx, len(subtask_list)):
                    # print(subtask_list[j][0])
                    if subtask_list[j][0] == 'OpenGripper':
                        frame_idx = subtask_list[j][1]
                        break
                seg_name = 'sequence_img_{:05d}.png'.format(frame_idx)
                # print(seg_name)
                seg_map = np.asarray(Image.open(os.path.join(segmentation_dir, seg_name)))
                tar_col = scene_struct['objects'][task_obj[0]]['segmentation_colour']
                tar_col = list(tar_col.values())[0:-1]
                tar_col = np.asarray(tar_col)
                tar_col = np.expand_dims(tar_col, axis=0)
                tar_col = np.expand_dims(tar_col, axis=0)
                # print(seg_map.shape)
                # print(tar_col.shape)
                diff = seg_map - tar_col
                diff = np.linalg.norm(diff, axis=2)
                diff = np.uint8(diff < 3)
                mask = coco.maskUtils.encode(np.asfortranarray(diff))
                mask['counts'] = str(mask['counts'], "utf-8")
                # im = Image.fromarray(np.uint8(diff)*255)
                # im.show()
                # Target from the final pos of the object - either last frame or last before default
                subtask_list_target.append(
                    [
                        sub_name,
                        sub_start,
                        sub_end,
                        task_obj[0],
                        mask
                    ]
                )
                mask_idxs.append(frame_idx)
            elif sub_name == 'MoveDOWN':
                # No target - maybe let it stay
                subtask_list_target.append(
                    [
                        sub_name,
                        sub_start,
                        sub_end,
                        None,
                        None
                    ]
                )
            elif sub_name == 'OpenGripper':
                # No target
                subtask_list_target.append(
                    [
                        sub_name,
                        sub_start,
                        sub_end,
                        None,
                        None
                    ]
                )
            elif sub_name == 'MoveToDefaultPos':
                # Sth like return above - target mb last mask???
                ob_idx = None
                mask = None
                for s in reversed(subtask_list_target):
                    if s[-1] is not None:
                        mask = s[-1]
                        ob_idx = s[-2]
                subtask_list_target.append(
                    [
                        sub_name,
                        sub_start,
                        sub_end,
                        ob_idx,
                        mask
                    ]
                )
                task_obj.pop(0)


        # print(mask_idxs)
        redu_seq_path = os.path.join(seq_dir, seq_name, 'data_interm.json')
        new_seq_path = os.path.join(seq_dir, seq_name, 'data_new.json')

        new_data = copy.deepcopy(data_f)
        for new_task in subtask_list_target:
            task_len = new_task[2] - new_task[1]
            for frame_id in range(new_task[1], new_task[2] + 1):
                progress = (frame_id - new_task[1]) / task_len
                new_data['sequence'][frame_id]['subtask_name'] = new_task[0]
                new_data['sequence'][frame_id]['subtask_start'] = new_task[1]
                new_data['sequence'][frame_id]['subtask_end'] = new_task[2]
                new_data['sequence'][frame_id]['subtask_object'] = new_task[3]
                new_data['sequence'][frame_id]['subtask_target_mask'] = new_task[4]
                new_data['sequence'][frame_id]['subtask_progress'] = progress
            # print(subtask_list_target)

        with open(new_seq_path, 'w') as f:
            json.dump(new_data, f, indent=4)

        new_data_redu = copy.deepcopy(new_data)
        new_redu_seq = [
            new_data['sequence'][frame_id] for frame_id in mask_idxs
        ]
        new_data_redu['sequence'] = new_redu_seq

        with open(redu_seq_path, 'w') as f:
            json.dump(new_data_redu, f, indent=4)

        # Parse task - multiple targets -mostly remove
        # Split subtask_list to groups
        # Match targets to groups
        # Fill up subtask targets
        # Maybe prep new data to generate segmentations without gripper









if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
