import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from pycocotools import coco


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', required=True)
parser.add_argument('--start_idx', '-si', type=int, default=0)
parser.add_argument('--end_idx', '-ei', type=int, default=-1)
parser.add_argument('--scene_json', default='SAMPLE_HD_train_')
parser.add_argument('--output_file', '-o', required=True)
args = parser.parse_args()


def main(args):
    seq_dir = os.path.join(args.input_dir, 'sequences')
    seq_names = sorted(os.listdir(seq_dir))
    start_idx = args.start_idx
    end_idx = args.end_idx
    seq_list = []
    seq_idxs = []
    for seq_name in seq_names:
        scene_num = int(seq_name[-6:])
        if scene_num < start_idx:
            continue
        if scene_num > end_idx:
            continue
        if scene_num in seq_idxs:
            continue
        seq_list.append(seq_name)
        seq_idxs.append(scene_num)

    object_list = []
    scene_dir = os.path.join(args.input_dir, 'scenes')
    tup_list = list(zip(seq_list, seq_idxs))
    for seq_name, seq_idx in tqdm(tup_list):
        instr_num = int(seq_name[1:7])
        scene_json = os.path.join(
            scene_dir, args.scene_json + '{:06d}.json'.format(seq_idx)
        )
        with open(scene_json, 'r') as f:
            scene_struct = json.load(f)
        segmentation_path = os.path.join(
            seq_dir, seq_name, 'segmentation'
        )
        seg_file = sorted(os.listdir(segmentation_path))[0]
        segmentation_path = os.path.join(segmentation_path, seg_file)
        seg_img = Image.open(segmentation_path)
        seg_img = np.asarray(seg_img)
        for idx, obj in enumerate(scene_struct['objects']):
            object_struct = {}
            seg_colour = obj['segmentation_colour']
            seg_colour = np.asarray(list(seg_colour.values())[:-1])
            seg_colour = np.expand_dims(seg_colour, 0)
            seg_colour = np.expand_dims(seg_colour, 0)
            diff = np.absolute(seg_img - seg_colour)
            diff = diff <= 1
            diff = np.all(diff, 2)
            diff = np.uint8(diff)
            mask = coco.maskUtils.encode(np.asfortranarray(diff))
            mask['counts'] = str(mask['counts'], "utf-8")
            object_struct['colour'] = obj['colour']
            object_struct['material'] = obj['material']
            object_struct['name'] = obj['name']
            if 'left' in obj['table_part']:
                object_struct['table_horizontal'] = 'left'
            elif 'right' in obj['table_part']:
                object_struct['table_horizontal'] = 'right'
            else:
                raise NotImplementedError()
            if 'front' in obj['table_part']:
                object_struct['table_vertical'] = 'front'
            elif 'back' in obj['table_part']:
                object_struct['table_vertical'] = 'back'
            else:
                raise NotImplementedError()
            # print(obj)
            object_struct['bounding_box'] = list(obj['bounding_box_size'].values())
            object_struct['bounding_box'] = [
                b * obj['scaling']['x'] for b in object_struct['bounding_box']
            ]
            object_struct['position'] = list(obj['position'].values())
            object_struct['mask'] = mask
            object_struct['scene_idx'] = seq_idx
            object_struct['obj_idx'] = idx
            object_struct['instr_idx'] = instr_num
            # print(object_struct)
            object_list.append(object_struct)

    with open(args.output_file, 'w') as f:
        json.dump(object_list, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
