from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader 
from sklearn.preprocessing import StandardScaler
import os, sys
import random
import itertools
import io
import math
import pandas as pd
import numpy as np
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from model import detr
from model import glpdepth


#키티
df = pd.read_csv('./annotations.csv')
train_image_list = os.listdir('./data/image/train')



