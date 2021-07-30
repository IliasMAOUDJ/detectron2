import PIL.Image
import numpy as np
import os.path
import json
import cv2
from imantics import Polygons, Mask
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import skimage.io as io
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

def gen_latim_dataset():
    for d in ["train", "val"]:
        if(d=="train"):
            DatasetCatalog.register("for_detectron_" + d, lambda d=d: get_latim_dicts("dataset/" + d))
            MetadataCatalog.get("for_detectron_" + d).set(thing_classes=["Femur", "Tibia", "Guide"])
        else:
            MetadataCatalog.get("for_detectron_" + d).set(thing_classes=["Femur", "Tibia", "Guide"], evaluator_type="coco")
            DatasetCatalog.register("for_detectron_" + d, lambda d=d: get_latim_dicts("dataset/" + d, real=True))