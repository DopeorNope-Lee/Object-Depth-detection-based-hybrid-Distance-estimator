# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 03:55:25 2022

@author: ODD Team
# GLPdepth을 이용해서 DETR에서 뽑은 BBOX의 DEPTH info 뽑아서 저장하기
"""

# 1. Import Module
import os, sys
import random
#import itertools
#import io
#import math
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import pandas as pd
import math
import time

from tqdm import tqdm

from torchvision import datasets, transforms
from transformers import GLPNForDepthEstimation, GLPNFeatureExtractor
from PIL import Image

from model.detr import DETR
from model.glpdepth import GLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################################################################
# 2. Data 및 변수 세팅
glp_kitti_preprocessing_data = pd.read_csv('./datasets/annotations.csv')
train_image_list = os.listdir('./datasets/data/image/train') # 7481

################################################################################################################################
# 3. Model 불러오기
# GLPdepth 불러오기
glp_pretrained = 'vinvino02/glpn-kitti'
GLPdepth = GLP(glp_pretrained)
GLPdepth.model.eval()

################################################################################################################################
# 4. Algorithm (Make data)
# 내가 원하는 이미지
'''
detr에서 얻은 KITTI와 동일한 객체 BBOX를 기준으로 그 안의 DEPTH 정보(min, mean)을 뽑아낸다.

min: GLPdepth은 거리가 가까울 수록 수치가 낮기 때문에 min을 이용해본다.
max: DPT은 거리가 가까울 수록 수치가 크기 때문에 max를 이용해본다.
'''

start = time.time() # 시간 측정 시작

# 데이터 프레임 저장할 곳
depth_mean = []
depth_min = []
depth_x = []
depth_y = []
depth_info = pd.DataFrame(columns={'depth_min','depth_mean','depth_x','depth_y'})

for k in range(len(train_image_list)): # 7481개의 데이터
    # 진행 상황 알라기
    print('이미지 전체 {} 중 {}번째 진행중'.format(len(train_image_list), k+1))
    
    # k번째 이미지
    filename = train_image_list[k]
    
    img = Image.open(os.path.join('./data/image/train/',filename))
    img_shape = cv2.imread(os.path.join('./data/image/train/',filename)).shape
    
    df_choose = glp_kitti_preprocessing_data[glp_kitti_preprocessing_data['filename']==filename]
    coordinates_array = df_choose[['xmin','ymin','xmax','ymax']].values
    
    # Make depth map
    prediction = GLPdepth.predict(img, img_shape)
    
    # append list
    for (xmin, ymin, xmax, ymax) in coordinates_array:
        depth_mean_info = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].mean()
        depth_min_info = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].min()
        xy = np.where(prediction==depth_min_info)

        depth_x.append(xy[1][0])
        depth_y.append(xy[0][0])
        depth_mean.append(depth_mean_info)
        depth_min.append(depth_min_info)

print('Finish')
end = time.time() # 시간 측정 끝
print(f"{end - start:.5f} sec") # 

# NA 값 확인
depth_info.isnull().sum(axis=0)
    
# 데이터 저장
depth_info['depth_mean'] = depth_mean
depth_info['depth_min'] = depth_min
depth_info['depth_x'] = depth_x
depth_info['depth_y'] = depth_y

# 데이터 병합
glp_kitti_preprocessing_data = pd.concat([glp_kitti_preprocessing_data, depth_info], axis=1)

# 데이터 저장 (최종)
glp_kitti_preprocessing_data.to_csv('./datasets/glp_kitti_data.csv', mode='a', index=False)