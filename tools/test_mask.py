import json
import os
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


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
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
trainer = DefaultTrainer(cfg)

from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
predictor = DefaultPredictor(cfg)
import cv2
import random
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode
for d in random.sample(val_dict, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=sample_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()


