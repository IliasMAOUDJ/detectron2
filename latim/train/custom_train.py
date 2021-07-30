#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
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
from detectron2.data import transforms as T

from detectron2.structures import BoxMode
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
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
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
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    augs = T.AugmentationList([
                            T.RandomBrightness(0.9, 1.1),
                            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                            T.RandomContrast(0.9,1.1)
                            ])  # type: T.Augmentation
    data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    #for d in ["train", "val"]:
    #    if(d=="train"):
    #        DatasetCatalog.register("for_detectron_synth_" + d, lambda d=d: get_latim_dicts("/home/ilias-m/Documents/DATASETS/LATIM/for_detectron_synth/" + d))
    #        MetadataCatalog.get("for_detectron_synth_" + d).set(thing_classes=["Femur", "Tibia", "Guide"])
    #    else:
    #        MetadataCatalog.get("for_detectron_synth_" + d).set(thing_classes=["Femur", "Tibia", "Guide"], evaluator_type="coco")
    #        DatasetCatalog.register("for_detectron_synth_" + d, lambda d=d: get_latim_dicts("/home/ilias-m/Documents/DATASETS/LATIM/for_detectron_synth/" + d, real=True))
#femur_tool_metadata = MetadataCatalog.get("for_detectron")  
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
