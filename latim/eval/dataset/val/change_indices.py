import os
from os import listdir
from os.path import isfile, join
import numpy as np

data = np.load('indices.npy')
data.sort()
print(data, len(data))