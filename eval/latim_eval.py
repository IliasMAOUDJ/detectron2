from detectron2.evaluation import COCOEvaluator, inference_on_dataset,print_csv_format
import os.path
import json
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data import transforms as T

from detectron2.structures import BoxMode

import detectron2.utils.comm as comm
import logging
logger = logging.getLogger("detectron2")

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
        split_file = os.path.join(img_dir, '{:s}'.format(indices_file))
        image_id = np.load(split_file)
        image_id =image_id
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
            for j in np.arange(1,len(bbox)+1):
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
    
import PIL.Image
def get_BB(img_filename):
    im = PIL.Image.open(img_filename)
    pixels = np.array(im.getdata()).reshape((im.size[1], im.size[0]))
    objects=[]
    for i in range(1,np.max(pixels)+1):
        if(i in pixels):
            objects.append([np.min(np.where(pixels == i)[1]),
                            np.min(np.where(pixels == i)[0]),
                            np.max(np.where(pixels == i)[1]),
                            np.max(np.where(pixels == i)[0])])
    return objects

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        print(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

import argparse
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.modeling import build_model

from collections import OrderedDict
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
import random    
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.data.datasets import register_coco_instances
from shutil import rmtree
from tqdm import tqdm
from detectron2.checkpoint import DetectionCheckpointer
if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    
    for d in ["val"]:
        MetadataCatalog.get("for_detectron_" + d).set(thing_classes=["Femur", "Tibia", "Guide"], evaluator_type="coco")
        DatasetCatalog.register("for_detectron_" + d, lambda d=d: get_latim_dicts("dataset/" + d, real=True))
    
    dir_name= cfg.MODEL.WEIGHTS.split(".")[0]
    rmtree("output")
    if os.path.exists(dir_name):
        rmtree(dir_name)
    os.mkdir(dir_name)
    gt_dir=dir_name+"/vis_gt/"
    os.mkdir(gt_dir)
    pred_dir=dir_name+"/vis_pred/"
    os.mkdir(pred_dir)
    
    femur_tool_metadata = MetadataCatalog.get("for_detectron_val")
    convert_to_coco_json("for_detectron_val", "output/output.json", allow_cached=False)
    dataset_dicts = get_latim_dicts("dataset/val", real=True)
    predictor = DefaultPredictor(cfg)
    for d in tqdm(dataset_dicts):  
        im = cv2.imread(d["file_name"])
        name = d["file_name"].split("/")[-1]
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=femur_tool_metadata)
        out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(pred_dir+name,out_pred.get_image()[:, :, ::-1])

        v = Visualizer(im[:, :, ::-1], metadata=femur_tool_metadata)
        out_gt = v.draw_dataset_dict(d)
        cv2.imwrite(gt_dir+name,out_gt.get_image()[:, :, ::-1])
    
    #
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    do_test(cfg, model) 

