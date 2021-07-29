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
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

from detectron2.structures import BoxMode
from imantics import Polygons, Mask
from detectron2.data import transforms as T

import sys
np.set_printoptions(threshold=sys.maxsize)
import skimage.io as io

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
        image_id =image_id[:30]
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


for d in ["train", "val"]:
    if(d=="train"):
        DatasetCatalog.register("for_detectron_synth_" + d, lambda d=d: get_latim_dicts("/home/ilias-m/Documents/DATASETS/LATIM/triplicated/" + d))
        MetadataCatalog.get("for_detectron_synth_" + d).set(thing_classes=["Femur", "Tibia", "Guide"])
    else:
        MetadataCatalog.get("for_detectron_synth_" + d).set(thing_classes=["Femur", "Tibia", "Guide"], evaluator_type="coco")
        DatasetCatalog.register("for_detectron_synth_" + d, lambda d=d: get_latim_dicts("/home/ilias-m/Documents/DATASETS/LATIM/triplicated/" + d, real=True))
femur_tool_metadata = MetadataCatalog.get("for_detectron_synth_val")

dataset_dicts = get_latim_dicts("/home/ilias-m/Documents/DATASETS/LATIM/no_tissue/val", real=True)



dataset = DatasetCatalog.get("for_detectron_synth_val")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=femur_tool_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("img", out.get_image()[:, :, ::-1])
    cv2.waitKey(0) 
  
    #closing all open windows 
    cv2.destroyAllWindows() 