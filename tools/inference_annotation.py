import setup

import os
import json
import argparse
import numpy as np
from config import get_cfg
from model import get_model
from tqdm import tqdm
from torchvision import transforms
from pycocotools import coco
from PIL import Image
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-cfg", default="default")
args = parser.parse_args()

cfg = get_cfg(args.config)

model = get_model(cfg)
model.set_test()
model.load_checkpoint(cfg.CONFIG.CHECKPOINT_PATH)
device = "cuda:0"

with open('/home/michas/Desktop/SAMPLE_HD/annotations_test.json') as f:
    annotations = json.load(f)

colour_dict = {
    "black": 0,
    "blue": 1,
    "brown": 2,
    "cyan": 3,
    "green": 4,
    "metallic": 5,
    "purple": 6,
    "red": 7,
    "transparent": 8,
    "white": 9,
    "yellow": 10
}

material_dict = {
    "ceramic": 0,
    "glass": 1,
    "metal": 2,
    "plastic": 3,
    "rubber": 4,
    "wooden": 5
}

name_dict = {
    "baking_tray": 0,
    "bowl": 1,
    "chopping_board": 2,
    "food_box": 3,
    "fork": 4,
    "glass": 5,
    "knife": 6,
    "mug": 7,
    "pan": 8,
    "plate": 9,
    "scissors": 10,
    "soda_can": 11,
    "spoon": 12,
    "thermos": 13,
    "wine_glass": 14
}

horizontal_dict = {
    'left': 0,
    'right': 1
}

vertical_dict = {
    'front': 0,
    'back': 1
}

inv_colour_dict = {v: k for k, v in colour_dict.items()}
inv_material_dict = {v: k for k, v in material_dict.items()}
inv_name_dict = {v: k for k, v in name_dict.items()}
inv_horizontal_dict = {v: k for k, v in horizontal_dict.items()}
inv_vertical_dict = {v: k for k, v in vertical_dict.items()}


predictions = []

transform_list = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((448, 448)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

seq_dir = '/home/michas/Desktop/SAMPLE_HD/sequences'

for ann in tqdm(annotations):
    obj = {}
    obj['scene_idx'] = ann['scene_idx']
    obj['obj_idx'] = ann['obj_idx']
    obj['instr_idx'] = ann['instr_idx']
    mask = coco.maskUtils.decode(ann['mask'])
    img_path = 'i{:06d}_s{:06d}'.format(ann["instr_idx"], ann["scene_idx"])
    # print(sum(sum(mask)))
    img_path = os.path.join(seq_dir, img_path, 'sequence_img_00000.jpg')
    # print(img_path)
    img = Image.open(img_path)
    img = np.asarray(img)
    # print(img.shape)
    img = np.concatenate((img, img * np.expand_dims(mask, 2)), 0)
    # img = img[:, :, [2, 1, 0]]
    img = transform_list(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    col_pred, mat_pred, name_pred, hor_pred, ver_pred, pos_pred, bb_pred = model.get_prediction(img)

    _, col_pred = torch.max(col_pred, dim=1)
    _, mat_pred = torch.max(mat_pred, dim=1)
    _, name_pred = torch.max(name_pred, dim=1)
    _, hor_pred = torch.max(hor_pred, dim=1)
    _, ver_pred = torch.max(ver_pred, dim=1)

    col_pred = inv_colour_dict[col_pred.cpu().numpy()[0]]
    mat_pred = inv_material_dict[mat_pred.cpu().numpy()[0]]
    name_pred = inv_name_dict[name_pred.cpu().numpy()[0]]
    hor_pred = inv_horizontal_dict[hor_pred.cpu().numpy()[0]]
    ver_pred = inv_vertical_dict[ver_pred.cpu().numpy()[0]]

    obj['colour'] = col_pred
    obj['material'] = mat_pred
    obj['name'] = name_pred
    obj['table_horizontal'] = hor_pred
    obj['table_vertical'] = ver_pred
    obj['bounding_box'] = bb_pred.detach().cpu().numpy().tolist()
    obj['position'] = pos_pred.detach().cpu().numpy().tolist()

    predictions.append(obj)

with open('/home/michas/Desktop/codes/nips2021/outputs/ann_res.json', 'w') as f:
    json.dump(predictions, f, indent=4)
















