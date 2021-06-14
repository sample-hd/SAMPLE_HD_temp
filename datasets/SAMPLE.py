import os
import h5py
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from pycocotools import coco
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class SAMPLE_Instructions(Dataset):
    def __init__(self, path, h5_instructions, vocab_json,
                 img_dir, img_prefix, idx_list=None):
        super(SAMPLE_Instructions, self).__init__()
        assert os.path.isfile(h5_instructions), "Instructions h5 missing"
        instructions_h5 = h5py.File(h5_instructions, 'r')
        self.instructions = np.asarray(
            instructions_h5['instructions'], dtype=np.int64)
        self.instruction_lengths = np.asarray(
            instructions_h5['instruction_lengths'], dtype=np.int64)
        self.instruction_idxs = np.asarray(
            instructions_h5['orig_idxs'], dtype=np.int64)
        self.image_idxs = np.asarray(
            instructions_h5['image_idxs'], dtype=np.int64)
        self.subtasks = np.asarray(
            instructions_h5['subtasks'], dtype=np.int64)
        self.subtask_lengths = np.asarray(
            instructions_h5['subtask_lengths'], dtype=np.int64)
        self.subtask_mask_sizes = np.asarray(
            instructions_h5['subtask_mask_sizes'], dtype=np.int64)
        self.subtask_mask_counts = np.asarray(
            instructions_h5['subtask_mask_counts'])

        # print("self.instruction_idxs")
        # print(self.instruction_idxs)

        if idx_list is not None:
            idx_arr = np.array(idx_list)
            # print(np.isin(idx_arr, self.instruction_idxs))
            assert np.all(np.isin(idx_arr, self.instruction_idxs)), "Incorrect idxs"
            # print(self.instruction_idxs[0:100])
            arr_idx_left = np.searchsorted(self.instruction_idxs, idx_arr)
            self.instruction_idxs = self.instruction_idxs[arr_idx_left]
            self.instructions = self.instructions[arr_idx_left]
            self.instruction_lengths = self.instruction_lengths[arr_idx_left]
            self.image_idxs = self.image_idxs[arr_idx_left]
            self.subtasks = self.subtasks[arr_idx_left]
            self.subtask_lengths = self.subtask_lengths[arr_idx_left]
            self.subtask_mask_sizes = self.subtask_mask_sizes[arr_idx_left]
            self.subtask_mask_counts = self.subtask_mask_counts[arr_idx_left]
            # print(self.instruction_idxs)

        self.img_dir = img_dir
        self.img_prefix = img_prefix

        self.img_transforms = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transforms = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(180)
        ])

        with open(vocab_json, 'r') as f:
            self.vocab = json.load(f)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        instruction_len = self.instruction_lengths[idx]
        subtask = self.subtasks[idx]
        subtask_len = self.subtask_lengths[idx]

        mask_sizes = self.subtask_mask_sizes[idx]
        mask_counts = self.subtask_mask_counts[idx]

        unq_rows = np.unique(mask_sizes, axis=0)
        unq_rows = unq_rows[~np.any(unq_rows < 0, axis=1)]
        assert unq_rows.shape[0] == 1
        mask_size = unq_rows[0]
        masks, mask_bools = self.get_mask_array(mask_counts, mask_size)

        instruction = torch.tensor(instruction)
        instruction_len = torch.tensor(instruction_len)
        subtask = torch.tensor(subtask)
        subtask_len = torch.tensor(subtask_len)
        masks = torch.tensor(masks, dtype=torch.float)
        mask_bools = torch.tensor(mask_bools)

        img_name = self.img_prefix + "{:06d}".format(self.image_idxs[idx]) + '.png'
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path)
        # img.show()
        img = self.img_transforms(img)

        masks = self.mask_transforms(masks)

        return instruction, instruction_len, subtask, subtask_len, masks, mask_bools, img

    def __str__(self):
        return "SAMPLE Instruction prediction dataset"

    def __len__(self):
        return len(self.instructions)

    def get_mask_array(self, mask_counts, mask_size):
        unique_counts = np.unique(mask_counts)
        # print(unique_counts)
        unique_dict = {}
        for count in unique_counts:
            if count == b'':
                arr = -np.ones(mask_size) * np.inf
                # print(arr.shape)
            else:
                mask = {
                    'size': mask_size,
                    'counts': count.decode()
                }
                arr = coco.maskUtils.decode(mask)
            unique_dict[count] = arr
        mask_arr = np.stack(
            [unique_dict[c] for c in mask_counts])
        # print(mask_arr.shape)
        mask_mask = np.array([c != b'' for c in mask_counts])
        # print(mask_mask

        return mask_arr, mask_mask

    def view_masks(self, idx):
        _, _, _, _, masks, mask_bools, _ = self.__getitem__(idx)
        masks = masks.numpy()
        mask_bools = mask_bools.numpy()
        # print(mask_bools.shape)
        img_size = [1600, 180 * (mask_bools.shape[0] // 5 + (mask_bools.shape[0] % 5 > 0))]
        img = Image.new("1", img_size)
        for i in range(mask_bools.shape[0]):
            col = i % 5
            row = i // 5
            # print(row, col)
            mask = masks[i]
            mask[mask == -np.inf] = 0
            im = Image.fromarray(np.uint8(mask) * 255)
            im = im.resize((320, 180))
            bg = Image.fromarray(np.uint8(255 * np.ones((180, 320))))
            # bg.show()
            im = im.crop((1, 1, 318, 178))
            bg.paste(im, (1, 1))
            # im.show()
            img.paste(bg, (col * 320, row * 180))
            # break
            # img.show()
        img.show()


class SAMPLE_sequence_reference(Dataset):
    def __init__(self, path, vocab_json, idx_list=None):
        super(SAMPLE_sequence_reference, self).__init__()

        seq_dir = os.path.join(path, 'sequences')
        sequence_dirlist = sorted(os.listdir(seq_dir))
        sequence_idxs = [
            int(s[1:7]) for s in sequence_dirlist
        ]
        seq_dict = dict(zip(sequence_idxs, sequence_dirlist))
        if idx_list is not None:
            assert set(idx_list).issubset(set(sequence_idxs)), "Wrong idxs"
            seq_dict = {k: seq_dict[k] for k in idx_list}

        self.seq_paths = {k: os.path.join(seq_dir, seq_dict[k]) for k in seq_dict.keys()}
        # print(self.seq_paths)
        self.sample_list = []
        for seq_idx, path in tqdm(self.seq_paths.items()):
            with open(os.path.join(path, 'data_subtasks.json'), 'r') as f:
                seq_data = json.load(f)['sequence']
            num_samples = len(seq_data)
            num_samples -= len([s for s in seq_data if s['subtask'] == "Succeeded"])
            num_samples -= len([s for s in seq_data if s['subtask'] == "Completed"])
            self.sample_list += list(zip([seq_idx] * num_samples, list(range(num_samples))))

        # print(self.sample_list)
        with open(vocab_json, 'r') as f:
            vocab = json.load(f)
        self.subtask_token_to_idx = vocab['subtask_token_to_idx']
        self.vocab = vocab

        self.img_transforms = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transforms = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(180)
        ])

    def __getitem__(self, idx):
        instr_idx, frame_num = self.sample_list[idx]
        # print(self.seq_paths[instr_idx])
        seq_path = self.seq_paths[instr_idx]
        data_file = os.path.join(seq_path, 'data_subtasks.json')
        with open(data_file, 'r') as f:
            seq_data = json.load(f)

        frame_data = seq_data['sequence'][frame_num]
        relative_frame = [
            f for f in seq_data['subtasks'] if frame_num >= f['subtask_start'] and frame_num <= f['subtask_end']
        ]
        # assert len(relative_frame) == 1, "Wrong subtask"
        if len(relative_frame) != 1:
            print(relative_frame)
            print(frame_data)
            print(seq_data['subtasks'])
        relative_frame = relative_frame[0]['frame_number']
        subtask_token = self.subtask_token_to_idx[frame_data['subtask']]

        """
        seq_img_file = "sequence_img_{:05d}.jpg".format(frame_num)
        seq_img_file = os.path.join(seq_path, seq_img_file)
        rel_img_file = "sequence_img_{:05d}.jpg".format(relative_frame)
        rel_img_file = os.path.join(seq_path, rel_img_file)

        mask = frame_data['subtask_target_mask']
        if mask is not None:
            mask = coco.maskUtils.decode(mask)
            mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
            mask = self.mask_transforms(mask)
        else:
            mask = self.get_blank_mask()

        progress = frame_data['subtask_progress']
        progress = torch.tensor(progress, dtype=torch.float)

        steering = frame_data['steering']
        steering = torch.tensor(steering, dtype=torch.float)

        seq_img = Image.open(seq_img_file)
        seq_img = self.img_transforms(seq_img)
        rel_img = Image.open(rel_img_file)
        rel_img = self.img_transforms(rel_img)
        # print(seq_img.shape)

        """
        # print(frame_data)
        # print(relative_frame)

        # print(frame_num)
        # print(seq_img_file)
        # print(rel_img_file)
        return None
        # return seq_img, rel_img, subtask_token, mask, progress, steering

    def __str__(self):
        return "SAMPLE single sequence images - steering + reference"

    def __len__(self):
        return len(self.sample_list)

    def get_blank_mask(self):
        return torch.zeros((180, 320), dtype=torch.float).unsqueeze(0)


class SAMPLE_effector_pos(Dataset):
    def __init__(self, path, h5_path=None, idx_list=None):
        super(SAMPLE_effector_pos, self).__init__()

        seq_dir = os.path.join(path, 'sequences')
        self.use_h5 = False
        if h5_path is not None:
            self.use_h5 = True

        if not self.use_h5:
            sequence_dirlist = sorted(os.listdir(seq_dir))
            sequence_idxs = [
                int(s[1:7]) for s in sequence_dirlist
            ]
            seq_dict = dict(zip(sequence_idxs, sequence_dirlist))
            if idx_list is not None:
                assert set(idx_list).issubset(set(sequence_idxs)), "Wrong idxs"
                seq_dict = {k: seq_dict[k] for k in idx_list}

            self.seq_paths = {k: os.path.join(seq_dir, seq_dict[k]) for k in seq_dict.keys()}
            # print(self.seq_paths)
            self.image_paths = []
            self.positions = []
            for seq_idx, path in tqdm(self.seq_paths.items()):
                with open(os.path.join(path, 'data_effector.json'), 'r') as f:
                    seq_data = json.load(f)['sequence']
                for frame in seq_data:
                    img_name = "sequence_img_{:05d}.jpg".format(
                        frame['frame_number']
                    )
                    img_path = os.path.join(path, img_name)
                    effector_pos = frame['effector_end']['position']
                    effector_pos = list(effector_pos.values())
                    self.image_paths.append(img_path)
                    self.positions.append(effector_pos)
            self.positions = np.asarray(self.positions)
            self.positions = torch.tensor(self.positions, dtype=torch.float)
        else:
            h5_data = h5py.File(h5_path, 'r')
            img_paths = list(np.asarray(
                h5_data['image_paths']))
            self.image_paths = [os.path.join(seq_dir, p.decode("utf-8")) for p in img_paths]
            positions = np.asarray(
                h5_data['positions'])
            self.positions = torch.tensor(positions, dtype=torch.float)

        # print(self.image_paths)
        # print(self.positions)

        self.img_transforms = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        position = self.positions[idx]
        # print(img_path)
        # print(position)
        seq_img = Image.open(img_path)
        seq_img = self.img_transforms(seq_img)

        return seq_img, position

    def show_path(self, idx):
        img_path = self.image_paths[idx]
        position = self.positions[idx]
        # print(img_path)
        # print(position)
        # seq_img = Image.open(img_path)
        # seq_img = self.img_transforms(seq_img)
        print(img_path)

        # return seq_img, position

    def __str__(self):
        return "SAMPLE sequence images to effector position"

    def __len__(self):
        return len(self.image_paths)


class SAMPLE_instruction_program(Dataset):
    def __init__(self, h5_path, vocab_json):
        super(SAMPLE_instruction_program, self).__init__()

        with open(vocab_json, 'r') as f:
            self.vocab = json.load(f)

        h5_data = h5py.File(h5_path, 'r')
        instructions = np.asarray(h5_data['instructions'], dtype=np.int64)
        self.instructions = torch.tensor(instructions, dtype=torch.long)
        instruction_lengths = np.asarray(h5_data['instruction_lengths'], dtype=np.int64)
        self.instruction_lengths = torch.tensor(instruction_lengths, dtype=torch.long)
        programs = np.asarray(h5_data['programs'], dtype=np.int64)
        self.programs = torch.tensor(programs, dtype=torch.long)
        program_lengths = np.asarray(h5_data['program_lengths'], dtype=np.int64)
        self.program_lengths = torch.tensor(program_lengths, dtype=torch.long)

    def __getitem__(self, idx):
        instr = self.instructions[idx]
        instr_len = self.instruction_lengths[idx]
        prog = self.programs[idx]
        prog_len = self.program_lengths[idx]
        return instr, instr_len, prog, prog_len

    def __str__(self):
        return "SAMPLE sequence intruction to sequence program"

    def __len__(self):
        return len(self.instructions)


class SAMPLE_attributes(Dataset):
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

    def __init__(self, path, dir_path):
        super(SAMPLE_attributes, self).__init__()
        with open(path, 'r') as f:
            self.annotations = json.load(f)

        self.transform_list = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((448, 448)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.seq_dir = os.path.join(dir_path, 'sequences')

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        # print(ann)
        colour = torch.tensor(self.colour_dict[ann['colour']])
        material = torch.tensor(self.material_dict[ann['material']])
        name = torch.tensor(self.name_dict[ann['name']])
        hor = torch.tensor(self.horizontal_dict[ann['table_horizontal']])
        ver = torch.tensor(self.vertical_dict[ann['table_vertical']])
        pos = torch.tensor(ann['position'])
        bb = torch.tensor(ann['bounding_box'])
        # print(colour, material, name, hor, ver)
        mask = coco.maskUtils.decode(ann['mask'])
        img_path = 'i{:06d}_s{:06d}'.format(ann["instr_idx"], ann["scene_idx"])
        # print(sum(sum(mask)))
        img_path = os.path.join(self.seq_dir, img_path, 'sequence_img_00000.jpg')
        # print(img_path)
        img = Image.open(img_path)
        img = np.asarray(img)
        # print(img.shape)
        img = np.concatenate((img, img * np.expand_dims(mask, 2)), 0)
        # img = img[:, :, [2, 1, 0]]
        img = self.transform_list(img)

        return img, colour, material, name, hor, ver, pos, bb

    def show_image(self, idx):
        img, _, _, _, _, _, _, _ = self.__getitem__(idx)
        # print(img.shape)
        # print(torch.tensor([0.229, 0.224, 0.225]).shape)
        img = img * torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
        img = img + torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        img = transforms.ToPILImage()(img)
        img.show()

    def __str__(self):
        return "SAMPLE attributes"

    def __len__(self):
        return len(self.annotations)








