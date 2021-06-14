import os


class Executor:
    def __init__(self, target_sep=['[', ']'], task_sep='_'):
        self.target_sep = target_sep
        self.task_sep = task_sep

    def parse_unity(self, sequence, scene_struct, idx=None):
        # unity_instr = {}
        # unity_instr['image_filename'] = scene_struct['image_filename']
        img_idx = os.path.splitext(scene_struct['image_filename'])[0]
        img_idx = int(img_idx[-6:])
        # print(img_idx)
        temp_targets = []
        obj_target = []
        task_name = None
        subject_obj = None
        object_obj = None
        subject_str = None

        if idx is None:
            idx = -1

        scene_objs = scene_struct['objects']

        for prog in sequence:
            if any([p in prog for p in ['place', 'stack', 'put']]):
                # print(prog)
                task = None
                sub = None
                if self.target_sep[0] in prog:
                    task, sub = prog.split(self.target_sep[0])
                    sub = sub.strip(self.target_sep[1])
                else:
                    task = prog

                if task_name is not None:
                    return None

                task_name = task
                subject_str = sub

            if prog == 'scene':
                obj_target += temp_targets
                # print(obj_target)
                temp_targets = list(range(len(scene_objs)))

            if 'filter' in prog:
                # print(prog.split(self.task_sep))
                prop, tar = prog.split(self.target_sep[0])
                tar = tar.strip(self.target_sep[1])
                prop = prop.split(self.task_sep)[1]
                # print(prop, tar)
                if prop == 'table':
                    temp_targets = self.filter_table(temp_targets, scene_objs, tar)
                elif prop == 'relate':
                    temp_targets = self.filter_relate(temp_targets, scene_objs, tar)
                    if temp_targets is None:
                        return None
                else:
                    temp_targets = self.filter_prop(temp_targets, scene_objs, prop, tar)

            # print(obj_target)
            # print(temp_targets)
        obj_target += temp_targets
        if task is None:
            return None
        if subject_str is None:
            subject_str = []
        else:
            subject_str = [subject_str]

        if task in ['stack', 'put']:
            if len(obj_target) > 2:
                return None
            else:
                object_obj = [obj_target[0]]
                subject_obj = obj_target[1]
        else:
            object_obj = obj_target
            subject_obj = -1

        unity_instr = {
            "image_filename": scene_struct['image_filename'],
            "image_index": img_idx,
            "instruction_idx": idx,
            "unity_program":
            {
                "name": task,
                "objectInt": object_obj,
                "subjectString": subject_str,
                "subjectInt": subject_obj
            }
        }

        return unity_instr

    def filter_table(self, obj_idxs, obj_list, tar):
        result = []
        for idx in obj_idxs:
            obj = obj_list[idx]
            if tar in obj['table_part']:
                result.append(idx)
        return result

    def filter_relate(self, obj_idxs, obj_list, tar):
        # result = [
        if len(obj_idxs) > 1:
            return None
        tar_obj = obj_list[obj_idxs[0]]
        # print(tar_obj['directions'][tar])
        return tar_obj['directions'][tar]

    def filter_prop(self, obj_idxs, obj_list, prop, tar):
        result = []
        for idx in obj_idxs:
            obj = obj_list[idx]
            if obj[prop] == tar:
                result.append(idx)
        return result

