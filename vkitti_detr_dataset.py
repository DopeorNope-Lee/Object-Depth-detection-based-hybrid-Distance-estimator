# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 03:08:25 2022

@author: Admin
"""

# 1. Import Module
import os
import numpy as np
import numpy
import torch
import cv2
import pandas as pd
import time
from tqdm import tqdm
from PIL import Image
from model.detr import DETR
from model.glpdepth import GLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################################################################
# 2. Data 및 변수 세팅
df = pd.read_csv('./datasets/vkitti_annotations.csv')
basic_path = './datasets/data/VKITTI/'
image_list = df['filename'].unique() # 2010

################################################################################################################################
# 3. Model 불러오기
# COCO classes (91개)
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# DETR 불러오기
model_path = 'facebookresearch/detr:main'
model_backbone = 'detr_resnet101'
#sys.modules.pop('models') # ModuleNotFoundError: No module named 'models.backbone' 이 에러 발생시 수행
DETR = DETR(model_path, model_backbone)
DETR.model.eval()
DETR.model.to(device)

################################################################################################################################
# 4. Algorithm (Make data)
# 좌표값의 제곱합(SSE)을 이용
'''
알고리즘의 목적: 실제 데이터에서 나타나는 bounding box보다 DETR이 예측하는 BOXES가 더 많아서, detr이 예측한 boxes중 어떤 것이 실제 데이터와 맞는지 비교해서 그 데이터의 zloc를 이용할 수 있게 하는 데이터 전처리 과정
Method: bounding box간의 좌표 제곱 합을 비교해서 가장 가깝다면 같은 객체를 인식한 bbox라고 판단한다. 단 하나의 객체를 DETR의 여러 BBOX가 같다고 판단했다면, 중복이기 때문에 그 데이터는 사용하지 않는다.
'''
start = time.time() # 시간 측정 시작

# 최종 df
detr_vkitti_preprocessing_data = pd.DataFrame()

# 내가 원하는 이미지
for k in tqdm(range(len(image_list))): # 2010개의 데이터
    # 진행 상황 알라기
    if k % 50 == 0:
        print('이미지 전체 {} 중 {}번째 진행중'.format(len(image_list), k+1))

    mask = df['filename'] == image_list[k]
    df_choose = df.loc[mask]
    
    # Real data의 class와 좌표값 
    class_list = df_choose[['class']].values
    coordinates = df_choose[['xmin','ymin','xmax','ymax']].values
    
    # Image Path
    name_split = image_list[k].split('_')
    scene = name_split[0] 
    img_name = name_split[1]+'_'+name_split[2]
    weather = df_choose['weather'].values[0]
    
    path = basic_path + scene + '/' + weather + '/' + 'frames/rgb/Camera_1/' + img_name 
    
    # 이미지 open and make Variable
    img = Image.open(path)
    img_shape = cv2.imread(path).shape
    
    # 예측
    scores, boxes = DETR.detect(img) # Detection
    
    boxes = boxes.cpu() # cpu로 전환
    
    input_coordinates = [] # DETR's bounding box
    label = [] # detr label
    
    count = boxes.shape[0]
    if count == 0:
        continue
    
    else:
        for (real_xmin, real_ymin, real_xmax, real_ymax) in coordinates.tolist():
            real_coord_array = np.repeat(np.array((real_xmin, real_ymin, real_xmax, real_ymax)).reshape(1,4), count, axis=0)
            
            result = np.sum(np.square(boxes.detach().numpy() - real_coord_array), axis=1) # 각각의 좌표를 빼서 가장 작은 값 찾기
            index = result.argmin()
            
            input_coordinates.append(boxes[index].detach().numpy())
            label.append(CLASSES[scores[index].argmax()])

    input_coordinates = np.array(input_coordinates)
    
    # 임의의 데이터 프레임 제작
    detr_df = pd.DataFrame({'filename':df_choose['filename'], 'class':label, 'real_class':df_choose['class'], 
                           'xmin':input_coordinates[:,0], 'ymin':input_coordinates[:,1], 'xmax':input_coordinates[:,2], 
                           'ymax':input_coordinates[:,3], 'angle':df_choose['angle'], 'zloc':df_choose['zloc'],
                           'weather':df_choose['weather']})

    #print(glp_df)
    # 형식에 맞게 class 조절
    for category in detr_df['class']:
        if category not in ['person', 'truck', 'car', 'bicycle', 'Misc', 'train']:
            detr_df['class'].replace({category:'Misc'}, inplace=True) # Misc class 설정
            
    # 중복 데이터 제거
    detr_df.drop_duplicates(['xmin','ymin','xmax','ymax'], inplace=True) # keep=False

    detr_df = detr_df.loc[detr_df['class']==detr_df['real_class']] # class가 다르면 제외하기
    detr_df.reset_index(inplace=True)
    detr_df.drop('index',inplace=True,axis=1)

    # 데이터 병합
    detr_vkitti_preprocessing_data = pd.concat([detr_vkitti_preprocessing_data, detr_df], axis=0)

print('Finish')
end = time.time() # 시간 측정 끝
print(f"{end - start:.5f} sec") # 186.10624 sec

# Information
print(detr_vkitti_preprocessing_data.isnull().sum(axis=0)) # NA 값 확인

# 최종 저장
detr_vkitti_preprocessing_data.drop('real_class', axis=1, inplace=True)

detr_vkitti_preprocessing_data.to_csv('./datasets/detr_vkitti_preprocessing_data.csv', mode='a', index=False) # csv 저장