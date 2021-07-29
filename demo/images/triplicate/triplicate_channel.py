import PIL.Image
import os
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sub_dirs_content = [files for path, subdirs, files in os.walk(ROOT_DIR)]

for file_list in sub_dirs_content:
    for file in (file_list):
        if("png" in file):
            I = np.asarray(PIL.Image.open(ROOT_DIR+"/"+file))
            im = PIL.Image.fromarray(I)
            w, h = im.size
            ima = Image.new('RGB', (w,h))
            data = zip(im.getdata(), im.getdata(), im.getdata())
            ima.putdata(list(data))
            imaga=np.asarray(ima)
            print(imaga.shape)

            ima.save(ROOT_DIR+"/triplicated_"+file)

