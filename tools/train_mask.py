import json
import os
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo

with open('/home/michas/Desktop/SAMPLE_HD/SAMPLE_scene_mask_train.json', 'r') as f:
    train_dict = json.load(f)
with open('/home/michas/Desktop/SAMPLE_HD/SAMPLE_scene_mask_test.json', 'r') as f:
    val_dict = json.load(f)


def get_dicts_train():
    return train_dict


def get_dicts_val():
    return val_dict


DatasetCatalog.register('sample_train', lambda: get_dicts_train())
DatasetCatalog.register('sample_val', lambda: get_dicts_val())

MetadataCatalog.get('sample_train').set(thing_classes=[
    "baking_tray", "bowl", "chopping_board", "food_box", "fork",
    "glass", "knife", "mug", "pan", "plate", "scissors", "soda_can",
    "spoon", "thermos", "wine_glass"])
MetadataCatalog.get('sample_val').set(thing_classes=[
    "baking_tray", "bowl", "chopping_board", "food_box", "fork",
    "glass", "knife", "mug", "pan", "plate", "scissors", "soda_can",
    "spoon", "thermos", "wine_glass"])

sample_metadata = MetadataCatalog.get("sample_train")
# print(sample_metadata)

# dataset_dicts = get_dicts_train()

# import random, cv2
# from detectron2.utils.visualizer import Visualizer
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=sample_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow('as', out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

# exit()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("sample_train",)
cfg.DATASETS.TEST = ("sample_val",)
cfg.DATALOADER.NUM_WORKERS = 12
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 60000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.OUTPUT_DIR = '../outputs/masktest'
cfg.SOLVER.STEPS = [35000, 50000]
cfg.INPUT.MASK_FORMAT = 'bitmask'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
