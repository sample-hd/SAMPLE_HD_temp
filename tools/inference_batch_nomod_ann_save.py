import setup
import json
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple
# import time
from model import get_model
from config import get_cfg
from datasets import get_dataloader
import argparse
from PIL import Image
from torchvision import transforms
import torch
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
import datetime
import os
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--idx', '-i', default=8353, type=int)
args = parser.parse_args()

# cfg = get_cfg(args.config)
# model = get_model(cfg)
# model.load_checkpoint(cfg.CONFIG.CHECKPOINT_PATH)
# device = "cuda:0"

start_idx = 7913
end_idx = 7913 + 448 - 1

start_idx = 8250
end_idx = 7913 + 448 - (448 - 337) - 1
end_idx = 8360

idxs = [7914, 7917, 7919, 7921, 7936, 7937, 7938, 7939, 7941, 7942, 7943, 7944, 7945, 7948, 7951, 7952, 7958, 7967, 7968, 7969, 7970, 7971, 7972, 7973, 7975, 7976, 7979, 7981, 7985, 7986, 7988, 7990, 7993, 7999, 8007, 8012, 8014, 8015, 8016, 8017, 8027, 8034, 8036, 8038, 8039, 8040, 8041, 8042, 8044, 8045, 8046, 8047, 8048, 8052, 8054, 8065, 8066, 8067, 8068, 8070, 8075, 8082, 8084, 8087, 8088, 8089, 8092, 8094, 8108, 8110, 8112, 8116, 8129, 8131, 8132, 8141, 8142, 8143, 8144, 8145, 8146, 8147, 8148, 8149, 8152, 8154, 8155, 8158, 8159, 8160, 8163, 8171, 8172, 8188, 8189, 8190, 8191, 8195, 8201, 8202, 8203, 8205, 8207, 8208, 8209, 8210, 8211, 8212, 8218, 8223, 8224, 8232, 8238, 8240, 8241, 8243, 8244, 8245, 8246, 8249, 8250, 8251, 8252, 8253, 8254, 8255, 8260, 8261, 8263, 8265, 8269, 8271, 8272, 8273, 8274, 8275, 8276, 8277, 8278, 8279, 8282, 8283, 8285, 8286, 8289, 8291, 8292, 8297, 8299, 8300, 8306, 8310, 8311, 8312, 8313, 8314, 8315, 8316, 8317, 8321, 8322, 8332, 8338, 8340, 8342, 8343, 8344, 8345, 8349, 8350, 8351, 8353, 8360]


idxs = [8353]

idxs = [args.idx]

img_path = "/home/michas/Desktop/SAMPLE_HD/exchange.jpg"
frame_path = "/home/michas/Desktop/SAMPLE_HD/frame.json"

instr_path = '/home/michas/Desktop/SAMPLE_HD/instructions/instructions_unityGT_filtered.json'
# print(datetime.date)
date = "{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(
    datetime.datetime.now().day,
    datetime.datetime.now().month,
    datetime.datetime.now().year % 100,
    datetime.datetime.now().hour,
    datetime.datetime.now().minute,
    datetime.datetime.now().second
)
# print(date)
res_path = '../outputs/inference_{}'.format(date)
res_json = os.path.join(res_path, 'results.json')

res_list = []

if not os.path.exists(res_path):
    os.makedirs(res_path)
# exit()

mid = [-0.431, 0.7589191]
left = [-0.381, -0.011]
right = [-0.861, -0.481]
front = [0.7789191, 0.9389191]
back = [0.5789191, 0.7389191]

mids = {
    "left": np.array([(left[1] + left[0]) / 2, (front[1] + back[0]) / 2]),
    "right": np.array([(right[1] + right[0]) / 2, (front[1] + back[0]) / 2]),
    "front": np.array([(left[1] + right[0]) / 2, (front[1] + front[0]) / 2]),
    "back": np.array([(left[1] + right[0]) / 2, (back[1] + back[0]) / 2]),
}


# print(mids)
z_hor = (front[1] - back[0]) / 2
z_vert = (front[1] - front[0]) / 2
x_vert = (left[1] - right[0]) / 2
x_hor = (left[1] - left[0]) / 2
max_r_h = np.sqrt(x_hor ** 2 + z_hor ** 2)
max_r_v = np.sqrt(x_vert ** 2 + z_vert ** 2)
max_r = max(max_r_h, max_r_v)
step = 0.025
r_values = np.arange(step, max_r, step)
# print(r_values)
# print("ADS")
placements_h = []
placements_v = []

for r in r_values:
    angle_inc = step / r
    angle_vals = np.arange(0, 2 * np.pi, angle_inc)
    sins = np.sin(angle_vals)
    coss = np.cos(angle_vals)
    points = np.stack((-sins, coss))
    points = r * points
    points = list(points.T)
    points_h = [p for p in points if p[0] < x_hor]
    # print(points_h)
    points_h = [p for p in points_h if p[0] > -x_hor]
    points_h = [p for p in points_h if p[1] > -z_hor]
    points_h = [p for p in points_h if p[1] < z_hor]
    points_v = [p for p in points if p[0] < x_vert]
    points_v = [p for p in points_v if p[0] > -x_vert]
    points_v = [p for p in points_v if p[1] > -z_vert]
    points_v = [p for p in points_v if p[1] < z_vert]
    placements_h += points_h
    placements_v += points_v


img_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def quaternion_inverse(quaternion):
    norm = np.sum(quaternion ** 2)
    quat = np.array([quaternion[0],
                     -quaternion[1],
                     -quaternion[2],
                     -quaternion[3]
                     ])
    return quat / norm


def quaternion_multiply(q1, q2):
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    mult = np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    ])
    return mult


for instr_idx in idxs:

    with open(instr_path, 'r') as f:
        target_instr = json.load(f)["instructions"][instr_idx]

    scene_json = "/home/michas/Desktop/codes/nips2021/outputs/scenes/SAMPLE_HD_train_{:06d}.json".format(
        target_instr["image_index"]
    )

    scene_json_unity = "/home/michas/Desktop/SAMPLE_HD/scenes/SAMPLE_HD_train_{:06d}.json".format(
        target_instr["image_index"]
    )

    info_dict = {
        "scene_json": scene_json_unity,
        "unity_instr_json": "/home/michas/Desktop/SAMPLE_HD/instructions/instructions_unityGT_filtered.json",
        "instr_idx": instr_idx,
    }

    with open("/home/michas/Desktop/SAMPLE_HD/exchange.json", 'w') as f:
        json.dump(info_dict, f)

    with open(scene_json, 'r') as f:
        scene = json.load(f)["objects"]
    with open(scene_json_unity, 'r') as f:
        scenegt = json.load(f)["objects"]

    objects = target_instr["unity_program"]["objectInt"]
    target = target_instr["unity_program"]["subjectInt"]
    if target < 0:
        target = target_instr["unity_program"]["subjectString"][0]

    # print(objects)
    # print(target)

    act_list = [
        "MoveOver",
        "MoveDown",
        "Close",
        "MoveUp",
        "MoveTarget",
        "MoveDownObj",
        "Release",
        "MoveUpDef"
    ]
    act_list = len(objects) * act_list
    act_list = act_list[:-1]
    # print(act_list)

    current_act = None
    countdown_val = 10
    # channel = EnvironmentParametersChannel()
    # channel.set_float_parameter("parameter_1", 2.0)
    # env = UnityEnvironment(file_name=None, seed=1)
    env = UnityEnvironment(file_name='/home/michas/Desktop/codes/nips2021/output/a/inference_build_test2.x86_64')
    # print("AAAAAAAAAAAAAAAAAAAAAAA")
    env.reset()
    # print("BBBBBBBBBBBBBBBBBBBBBBBBBBB")
    behavior_names = env.behavior_specs.keys()
    b_name = list(behavior_names)[0]
    action = ActionTuple(np.array([[0, 0, 0, 0, 0, 0, 1]]))
    # while True:
    # env.set_actions(b_name, action)
    # env = UnityEnvironment(side_channels=[channel])
    steer_gripper = 1
    countdown = countdown_val
    current_targ = objects.pop(0)
    target_orientation = None
    success = False
    pos_save = None
    pos_counter = 0
    with torch.no_grad():
        for i in range(10000):
            frame_img = '/home/michas/Desktop/SAMPLE_HD/results/sequence_img_{:06d}'.format(i)
            shutil.copyfile(img_path, frame_img)
            pos_counter += 1
            # env.close()
            # break
            # print('dsfgfdsgfdsfgd')
            d, t = env.get_steps(b_name)
            # print(len(d))
            if len(d.obs[0]) > 0:
                if (d.obs[0][0][0] > 0):
                    success = True
                    # env.close()
                    break
                elif (d.obs[0][0][0] > 0):
                    success = False
                    # env.close()
                    break
            else:
                # env.close()
                break
            # if len(t[0]) > 0:
            #     print("t[0]")
            #     print(t[0])
            steer_x = 0
            steer_y = 0
            steer_z = 0
            steer_rot_x = 0
            steer_rot_y = 0
            steer_rot_z = 0
            if current_act is None:
                # print(current_act)
                if len(act_list) < 1:
                    current_act = None
                    break
                else:
                    current_act = act_list.pop(0)
                cd_done = False
                if current_act == "MoveOver":
                    target_pos = scene[current_targ]["position"]
                    target_box = scene[current_targ]["bounding_box_size"]
                    target_box = scenegt[current_targ]["bounding_box_size"]
                    scale = scenegt[current_targ]["scaling"]['x']
                    # scale = 1
                    for k, v in target_box.items():
                        target_box[k] = v * scale
                    current_pos_targ = None
                    current_pos_targ = list(target_pos.values())
                    # print(target_box)
                    if target_box['x'] < 0.08 and target_box['z'] < 0.08:
                        # print("MIGFGDI")
                        lower_down = 0.01
                    else:
                        current_pos_targ[0] -= (target_box['x'] * 0.5 - 0.0075)
                        lower_down = min(0.02, target_box['y'] - 0.003)
                    height_diff = 0.15
                    current_pos_targ[1] += (target_box['y'] - lower_down + height_diff)
                    current_pos_targ = np.array(current_pos_targ)
                elif current_act == "MoveDown":
                    current_pos_targ[1] = current_pos_targ[1] - height_diff
                elif current_act == "MoveDownObj":
                    # print("dnskfn")
                    current_pos_targ[1] = current_pos_targ[1] - (height_diff - 0.05)
                elif current_act == "MoveUp":
                    # current_pos_targ[1] = current_pos_targ[1] + height_diff
                    current_pos_targ = effector_pos
                    current_pos_targ[1] = current_pos_targ[1] + height_diff
                elif current_act == "MoveUpDef":
                    current_pos_targ[1] = current_pos_targ[1] + (height_diff - 0.05)
                    current_targ = objects.pop(0)
                elif current_act == "MoveTarget":
                    if isinstance(target, int):
                        target_pos = scene[target]["position"]
                        current_pos_targ[0] = target_pos['x']
                        current_pos_targ[2] = target_pos['z']
                    else:
                        obstacles = []
                        for o_id, o in enumerate(scene):
                            if target not in o['table_part']:
                                continue
                            pos = o['position']
                            mid = np.array([pos['x'], pos['z']])
                            bbox = o['bounding_box_size']
                            bbox = scenegt[o_id]['bounding_box_size']
                            x_half = bbox['x'] / 2 + 0.005
                            z_half = bbox['z'] / 2 + 0.005
                            orient = scenegt[o_id]['orientation']
                            orient = list(orient.values())
                            w = orient[0]
                            orient[0:3] = orient[1:4]
                            orient[3] = w
                            rot = R.from_quat(orient)
                            rot = rot.as_euler('xyz', degrees=False)
                            if np.abs(rot[0]) > np.abs(rot[2]):
                                rot = - rot[1]
                            else:
                                rot = rot[1]
                            s = np.sin(rot)
                            c = np.cos(rot)
                            verts = [
                                np.array([x_half * c - z_half * s, x_half * s + z_half * c]),
                                np.array([- x_half * c - z_half * s, - x_half * s + z_half * c]),
                                np.array([- x_half * c + z_half * s, - x_half * s - z_half * c]),
                                np.array([x_half * c + z_half * s, x_half * s - z_half * c])
                            ]
                            verts = [v + mid for v in verts]
                            obstacles.append(Polygon(verts))
                        if target in ['left', 'right']:
                            placements = placements_h
                        else:
                            placements = placements_v

                        obj = scene[current_targ]
                        # print(obj['prefab'])
                        bbox = obj['bounding_box_size']
                        bbox = scenegt[o_id]['bounding_box_size']
                        x_half = bbox['x'] / 2
                        z_half = bbox['z'] / 2
                        orient = scenegt[current_targ]['orientation']
                        orient = list(orient.values())
                        w = orient[0]
                        orient[0:3] = orient[1:4]
                        orient[3] = w
                        rot = R.from_quat(orient)
                        rot = rot.as_euler('xyz', degrees=False)
                        if np.abs(rot[0]) > np.abs(rot[2]):
                            rot = - rot[1]
                        else:
                            rot = rot[1]
                        s = np.sin(rot)
                        c = np.cos(rot)
                        verts = [
                            np.array([x_half * c - z_half * s, x_half * s + z_half * c]),
                            np.array([- x_half * c - z_half * s, - x_half * s + z_half * c]),
                            np.array([- x_half * c + z_half * s, - x_half * s - z_half * c]),
                            np.array([x_half * c + z_half * s, x_half * s - z_half * c])
                        ]
                        good_placement = None
                        min_overlap = 1000.0
                        for plac in placements:
                            test_point = mids[target] + plac
                            obj_verts = [v + test_point for v in verts]
                            test_poly = Polygon(obj_verts)
                            # plt.plot(*test_poly.exterior.xy)
                            overlap = 0
                            for poly in obstacles:
                                overlap += test_poly.intersection(poly).area
                            if overlap < min_overlap:
                                min_overlap = overlap
                                good_placement = test_point
                            if min_overlap < 0.0001:
                                break
                        current_pos_targ[0] = good_placement[0]
                        current_pos_targ[2] = good_placement[1]
                        scene[current_targ]['position']['x'] = current_pos_targ[0]
                        scene[current_targ]['position']['z'] = current_pos_targ[2]
                        scene[current_targ]['table_part'].append(target)

            with open(frame_path, 'r') as f:
                frame_info = json.load(f)
            if current_act in ["MoveOver", "MoveDown", "MoveDownObj", "MoveUp", "MoveTarget", "MoveUpDef"]:
                effector_pos = frame_info['effectorCoords']['position']
                effector_pos = np.array(list(effector_pos.values()))
                if pos_save is None:
                    pos_save = effector_pos
                if np.linalg.norm(effector_pos - pos_save) > 0.002:
                    pos_save = effector_pos
                    pos_counter = 0
                if (pos_counter > 50):
                    # env.close()
                    break
                # img_o = Image.open(img_path)
                # img = img_transforms(img_o)
                # img_o.close()
                # img = img.to(device)
                # img = img.unsqueeze(0)
                # pos_pred = model.get_prediction(img).squeeze()
                # print(pos_pred, effector_pos)
                # pos_pred = pos_pred.cpu().numpy()
                # effector_pos = pos_pred

                effector_rot = np.array(list(frame_info['effectorCoords']['orientation'].values()))
                pos_diff = effector_pos - current_pos_targ
                if target_orientation is None:
                    target_orientation = effector_rot

                steer_rot_x = 0
                steer_rot_y = 0

                if current_act in ['MoveOver', 'MoveTarget', 'MoveDown']:
                    rot_diff = quaternion_multiply(
                        target_orientation,
                        quaternion_inverse(effector_rot)
                    )
                    # print("rot_diff")
                    # print(rot_diff)
                    w = rot_diff[0]
                    rot_diff[0:3] = rot_diff[1:4]
                    rot_diff[3] = w
                    # print(rot_diff)
                    rot_euler = R.from_quat(rot_diff)
                    rot_euler = rot_euler.as_euler('xyz', degrees=True)
                    diff_x = (rot_euler[0] + 540) % 360 - 180
                    diff_z = (rot_euler[2] + 540) % 360 - 180

                    # print(diff_x)
                    # print(diff_z)

                    steer_rot_x = np.clip(diff_x / 2, -1, 1)
                    if np.absolute(steer_rot_x) < 0.01:
                        steer_rot_x = 0

                    steer_rot_y = np.clip(diff_z / 2, -1, 1)
                    if np.absolute(steer_rot_y) < 0.01:
                        steer_rot_y = 0

                if np.linalg.norm(pos_diff) > 0.005:
                    # if pos_diff[0] > 0.001:
                    steer_y = np.clip(-pos_diff[0] / 0.01, -1, 1)
                    # if pos_diff[2] > 0.001:
                    steer_x = np.clip(pos_diff[2] / 0.01, -1, 1)
                    # if pos_diff[1] > 0.001:
                    steer_z = np.clip(-pos_diff[1] / 0.01, -1, 1)
                else:
                    steer_x = 0
                    steer_y = 0
                    steer_z = 0
                    current_act = None
                if current_act == "MoveUp" and not cd_done:
                    countdown -= 2
                    steer_rot_x = 0
                    steer_rot_y = 0
                    steer_x = 0
                    steer_y = 0
                    if countdown < 0:
                        cd_done = True
                        countdown = countdown_val
            elif current_act in ["Close"]:
                steer_gripper = -1
                countdown -= 1
                if countdown < 0:
                    countdown = countdown_val
                    current_act = None
            elif current_act in ["Release"]:
                steer_gripper = 1
                countdown -= 1
                if countdown < 0:
                    countdown = countdown_val
                    current_act = None

            steering = np.array([[
                steer_x, steer_y, steer_z,
                steer_rot_x, steer_rot_y, steer_rot_z,
                steer_gripper
            ]])

            # print(steering)
            action = ActionTuple(steering)
            env.set_actions(b_name, action)
            env.step()
    env.close()
    print(success)
    res_list.append((instr_idx, success))
    with open(res_json, 'w') as f:
        json.dump(res_list, f)


