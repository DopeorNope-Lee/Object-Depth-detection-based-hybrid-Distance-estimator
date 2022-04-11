"""
    @ Author :ODD_team
"""

from sklearn import linear_model
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader 
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from transformers import GLPNForDepthEstimation, GLPNFeatureExtractor
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from transformers import GLPNForDepthEstimation, GLPNFeatureExtractor
from PIL import Image
import cv2






"""

# 모델 불러오기
DE_PATH="aaaa"
model = linear_model()
model.load_state_dict(torch.load('DE_PATH'))
model.eval()
"""