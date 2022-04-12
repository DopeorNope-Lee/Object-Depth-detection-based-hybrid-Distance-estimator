# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:33:23 2022

@author: Admin
"""

# 1. Import Module
import os
import pandas as pd
import numpy as np
import numpy
import torch
import cv2
import time
from tqdm import tqdm
from PIL import Image
from model.glpdepth import GLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################################################################
# 2. Data 및 변수 세팅
glp_vkitti_preprocessing_data = pd.read_csv('./datasets/detr_vkitti_preprocessing_data.csv')
basic_path = './datasets/data/VKITTI/'
image_list = glp_vkitti_preprocessing_data['filename'].unique() # 2010

################################################################################################################################
# 3. Model 불러오기
# GLPdepth 불러오기
glp_pretrained = 'vinvino02/glpn-kitti'
GLPdepth = GLP(glp_pretrained)
GLPdepth.model.eval()
GLPdepth.model.to(device)

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

for k in tqdm(range(len(image_list))): # 2021개의 데이터
    # 진행 상황 알라기
    if k % 50 == 0:
        print('이미지 전체 {} 중 {}번째 진행중'.format(len(image_list), k+1))
    
    # choose df
    mask = glp_vkitti_preprocessing_data['filename'] == image_list[k]
    df_choose = glp_vkitti_preprocessing_data.loc[mask]
    
    # Image Path
    name_split = image_list[k].split('_')
    scene = name_split[0] 
    img_name = name_split[1]+'_'+name_split[2]
    weather = df_choose['weather'].values[0]
    
    path = basic_path + scene + '/' + weather + '/' + 'frames/rgb/Camera_1/' + img_name 
    
    # 이미지 open and make Variable
    img = Image.open(path)
    img_shape = cv2.imread(path).shape
    
    coordinates_array = df_choose[['xmin','ymin','xmax','ymax']].values
    
    # Make depth map
    prediction = GLPdepth.predict(img, img_shape)
    # append list

    for (xmin, ymin, xmax, ymax) in coordinates_array:
        
        # depth map의 index는 최소한 0
        if int(xmin) < 0:
            xmin = 0
        if int(ymin) < 0:
            ymin = 0
        
        depth_mean_info = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].mean()
        depth_min_info = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].min()
        xy = np.where(prediction==depth_min_info)

        depth_x.append(xy[1][0])
        depth_y.append(xy[0][0])
        depth_mean.append(depth_mean_info)
        depth_min.append(depth_min_info)
        
# Check time
print('Finish')
end = time.time() # 시간 측정 끝
print(f"{end - start:.5f} sec") # 243.08134 sec

# 인덱스 재설정
glp_vkitti_preprocessing_data.reset_index(inplace=True)
glp_vkitti_preprocessing_data.drop('index', axis=1, inplace=True)

# 데이터 저장
depth_info['depth_mean'] = depth_mean
depth_info['depth_min'] = depth_min
depth_info['depth_x'] = depth_x
depth_info['depth_y'] = depth_y

# NA 값 확인
depth_info.isnull().sum(axis=0)

# 데이터 병합
glp_vkitti_preprocessing_data = pd.concat([glp_vkitti_preprocessing_data, depth_info], axis=1)

# 데이터 저장 (최종)
glp_vkitti_preprocessing_data.to_csv('./datasets/glp_vkitti_data.csv', mode='a', index=False)