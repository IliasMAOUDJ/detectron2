import logging
import os
from collections import OrderedDict
from matplotlib.pyplot import cla
from skimage import io
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper

from detectron2.structures import BoxMode
from imantics import Polygons, Mask
from detectron2.data import transforms as T

import sys
np.set_printoptions(threshold=sys.maxsize)

CATEGORY_ID = {"Femur": 0, "Tibia": 1, "Guide": 2}
def get_latim_dicts(img_dir, real=False):
    dataset_dicts = []
    if(real==True):
        json_file = os.path.join(img_dir, "real_data_annot_json.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        annos = list(imgs_anns.values())
        annos = [v for v in annos if v["regions"]]
        for idx, v in enumerate(annos):
            record = {}
            filename = os.path.join(img_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            objs = []
            anno = [r['shape_attributes'] for r in v['regions']]
            class_ids = [r['region_attributes'] for r in v['regions']]
            
            for region, class_id in zip(anno, class_ids):
                px = region["all_points_x"]
                py = region["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": CATEGORY_ID[class_id["type"]],
                }
                objs.append(obj)
                    
            record["annotations"] = objs
            dataset_dicts.append(record)
    else:
        indices_file = os.path.join(img_dir, "indices.npy")
        image_id = np.load(indices_file)
        for i in image_id:
            record = {}
            filename = os.path.join(img_dir, '{:06d}.png'.format(i))
            height, width = cv2.imread(filename).shape[:2]
            class_ids = np.load(os.path.join(img_dir, "../labels",'{:06d}.npy'.format(i)))
            record["file_name"] = filename
            record["image_id"] = i
            record["height"] = height
            record["width"] = width
            objs = []
            masks = os.path.join(img_dir, "../semantic_masks/",'{:06d}.png'.format(i))
            all_masks = io.imread(masks)
            bbox = get_BB(masks)
            cnt=0
            for j in np.arange(1,len(all_masks)+1):
                if(j not in all_masks):
                    continue
                mask = np.asarray(all_masks==j, dtype= np.uint8)
                polygons = Mask(mask).polygons()
                polygons = [p for x in polygons for p in x]
                obj = {
                    "bbox": bbox[cnt],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [polygons],
                    "category_id": j-1,
                }       
                cnt +=1
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts