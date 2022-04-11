# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 03:55:25 2022

@author: ODD Team
# DETR과 GLPdepth를 활용한 데이터 만들기
"""

# 1. Import Module
import os
import pandas as pd
import torch
import time

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################################################################
# 2. Data 및 변수 세팅
# scene1
scene1_clone = os.listdir('./VKITTI/Scene01/clone/frames/rgb/Camera_1/')
scene1_clone_bbox = pd.read_csv('./VKITTI_txt/Scene01/clone/bbox.txt', delimiter=' ')
scene1_clone_info = pd.read_csv('./VKITTI_txt/Scene01/clone/info.txt', delimiter=' ')
scene1_clone_pose = pd.read_csv('./VKITTI_txt/Scene01/clone/pose.txt', delimiter=' ')

scene1_fog = os.listdir('./VKITTI/Scene01/fog/frames/rgb/Camera_1/')
scene1_fog_bbox = pd.read_csv('./VKITTI_txt/Scene01/fog/bbox.txt', delimiter=' ')
scene1_fog_info = pd.read_csv('./VKITTI_txt/Scene01/fog/info.txt', delimiter=' ')
scene1_fog_pose = pd.read_csv('./VKITTI_txt/Scene01/fog/pose.txt', delimiter=' ')

scene1_morning = os.listdir('./VKITTI/Scene01/morning/frames/rgb/Camera_1/')
scene1_morning_bbox = pd.read_csv('./VKITTI_txt/Scene01/morning/bbox.txt', delimiter=' ')
scene1_morning_info = pd.read_csv('./VKITTI_txt/Scene01/morning/info.txt', delimiter=' ')
scene1_morning_pose = pd.read_csv('./VKITTI_txt/Scene01/morning/pose.txt', delimiter=' ')

scene1_overcast = os.listdir('./VKITTI/Scene01/overcast/frames/rgb/Camera_1/')
scene1_overcast_bbox = pd.read_csv('./VKITTI_txt/Scene01/overcast/bbox.txt', delimiter=' ')
scene1_overcast_info = pd.read_csv('./VKITTI_txt/Scene01/overcast/info.txt', delimiter=' ')
scene1_overcast_pose = pd.read_csv('./VKITTI_txt/Scene01/overcast/pose.txt', delimiter=' ')

scene1_rain = os.listdir('./VKITTI/Scene01/rain/frames/rgb/Camera_1/')
scene1_rain_bbox = pd.read_csv('./VKITTI_txt/Scene01/rain/bbox.txt', delimiter=' ')
scene1_rain_info = pd.read_csv('./VKITTI_txt/Scene01/rain/info.txt', delimiter=' ')
scene1_rain_pose = pd.read_csv('./VKITTI_txt/Scene01/rain/pose.txt', delimiter=' ')

scene1_sunset = os.listdir('./VKITTI/Scene01/sunset/frames/rgb/Camera_1/')
scene1_sunset_bbox = pd.read_csv('./VKITTI_txt/Scene01/sunset/bbox.txt', delimiter=' ')
scene1_sunset_info = pd.read_csv('./VKITTI_txt/Scene01/sunset/info.txt', delimiter=' ')
scene1_sunset_pose = pd.read_csv('./VKITTI_txt/Scene01/sunset/pose.txt', delimiter=' ')

# scene2
scene2_clone = os.listdir('./VKITTI/Scene02/clone/frames/rgb/Camera_1/')
scene2_clone_bbox = pd.read_csv('./VKITTI_txt/Scene02/clone/bbox.txt', delimiter=' ')
scene2_clone_info = pd.read_csv('./VKITTI_txt/Scene02/clone/info.txt', delimiter=' ')
scene2_clone_pose = pd.read_csv('./VKITTI_txt/Scene02/clone/pose.txt', delimiter=' ')

scene2_fog = os.listdir('./VKITTI/Scene02/fog/frames/rgb/Camera_1/')
scene2_fog_bbox = pd.read_csv('./VKITTI_txt/Scene02/fog/bbox.txt', delimiter=' ')
scene2_fog_info = pd.read_csv('./VKITTI_txt/Scene02/fog/info.txt', delimiter=' ')
scene2_fog_pose = pd.read_csv('./VKITTI_txt/Scene02/fog/pose.txt', delimiter=' ')

scene2_morning = os.listdir('./VKITTI/Scene02/morning/frames/rgb/Camera_1/')
scene2_morning_bbox = pd.read_csv('./VKITTI_txt/Scene02/morning/bbox.txt', delimiter=' ')
scene2_morning_info = pd.read_csv('./VKITTI_txt/Scene02/morning/info.txt', delimiter=' ')
scene2_morning_pose = pd.read_csv('./VKITTI_txt/Scene02/morning/pose.txt', delimiter=' ')

scene2_overcast = os.listdir('./VKITTI/Scene02/overcast/frames/rgb/Camera_1/')
scene2_overcast_bbox = pd.read_csv('./VKITTI_txt/Scene02/overcast/bbox.txt', delimiter=' ')
scene2_overcast_info = pd.read_csv('./VKITTI_txt/Scene02/overcast/info.txt', delimiter=' ')
scene2_overcast_pose = pd.read_csv('./VKITTI_txt/Scene02/overcast/pose.txt', delimiter=' ')

scene2_rain = os.listdir('./VKITTI/Scene02/rain/frames/rgb/Camera_1/')
scene2_rain_bbox = pd.read_csv('./VKITTI_txt/Scene02/rain/bbox.txt', delimiter=' ')
scene2_rain_info = pd.read_csv('./VKITTI_txt/Scene02/rain/info.txt', delimiter=' ')
scene2_rain_pose = pd.read_csv('./VKITTI_txt/Scene02/rain/pose.txt', delimiter=' ')

scene2_sunset = os.listdir('./VKITTI/Scene02/sunset/frames/rgb/Camera_1/')
scene2_sunset_bbox = pd.read_csv('./VKITTI_txt/Scene02/sunset/bbox.txt', delimiter=' ')
scene2_sunset_info = pd.read_csv('./VKITTI_txt/Scene02/sunset/info.txt', delimiter=' ')
scene2_sunset_pose = pd.read_csv('./VKITTI_txt/Scene02/sunset/pose.txt', delimiter=' ')

# scene 6
scene6_clone = os.listdir('./VKITTI/Scene06/clone/frames/rgb/Camera_1/')
scene6_clone_bbox = pd.read_csv('./VKITTI_txt/Scene06/clone/bbox.txt', delimiter=' ')
scene6_clone_info = pd.read_csv('./VKITTI_txt/Scene06/clone/info.txt', delimiter=' ')
scene6_clone_pose = pd.read_csv('./VKITTI_txt/Scene06/clone/pose.txt', delimiter=' ')

scene6_fog = os.listdir('./VKITTI/Scene06/fog/frames/rgb/Camera_1/')
scene6_fog_bbox = pd.read_csv('./VKITTI_txt/Scene06/fog/bbox.txt', delimiter=' ')
scene6_fog_info = pd.read_csv('./VKITTI_txt/Scene06/fog/info.txt', delimiter=' ')
scene6_fog_pose = pd.read_csv('./VKITTI_txt/Scene06/fog/pose.txt', delimiter=' ')

scene6_morning = os.listdir('./VKITTI/Scene06/morning/frames/rgb/Camera_1/')
scene6_morning_bbox = pd.read_csv('./VKITTI_txt/Scene06/morning/bbox.txt', delimiter=' ')
scene6_morning_info = pd.read_csv('./VKITTI_txt/Scene06/morning/info.txt', delimiter=' ')
scene6_morning_pose = pd.read_csv('./VKITTI_txt/Scene06/morning/pose.txt', delimiter=' ')

scene6_overcast = os.listdir('./VKITTI/Scene06/overcast/frames/rgb/Camera_1/')
scene6_overcast_bbox = pd.read_csv('./VKITTI_txt/Scene06/overcast/bbox.txt', delimiter=' ')
scene6_overcast_info = pd.read_csv('./VKITTI_txt/Scene06/overcast/info.txt', delimiter=' ')
scene6_overcast_pose = pd.read_csv('./VKITTI_txt/Scene06/overcast/pose.txt', delimiter=' ')

scene6_rain = os.listdir('./VKITTI/Scene06/rain/frames/rgb/Camera_1/')
scene6_rain_bbox = pd.read_csv('./VKITTI_txt/Scene06/rain/bbox.txt', delimiter=' ')
scene6_rain_info = pd.read_csv('./VKITTI_txt/Scene06/rain/info.txt', delimiter=' ')
scene6_rain_pose = pd.read_csv('./VKITTI_txt/Scene06/rain/pose.txt', delimiter=' ')

scene6_sunset = os.listdir('./VKITTI/Scene06/sunset/frames/rgb/Camera_1/')
scene6_sunset_bbox = pd.read_csv('./VKITTI_txt/Scene06/sunset/bbox.txt', delimiter=' ')
scene6_sunset_info = pd.read_csv('./VKITTI_txt/Scene06/sunset/info.txt', delimiter=' ')
scene6_sunset_pose = pd.read_csv('./VKITTI_txt/Scene06/sunset/pose.txt', delimiter=' ')

# scene 18
scene18_clone = os.listdir('./VKITTI/Scene18/clone/frames/rgb/Camera_1/')
scene18_clone_bbox = pd.read_csv('./VKITTI_txt/Scene18/clone/bbox.txt', delimiter=' ')
scene18_clone_info = pd.read_csv('./VKITTI_txt/Scene18/clone/info.txt', delimiter=' ')
scene18_clone_pose = pd.read_csv('./VKITTI_txt/Scene18/clone/pose.txt', delimiter=' ')

scene18_fog = os.listdir('./VKITTI/Scene18/fog/frames/rgb/Camera_1/')
scene18_fog_bbox = pd.read_csv('./VKITTI_txt/Scene18/fog/bbox.txt', delimiter=' ')
scene18_fog_info = pd.read_csv('./VKITTI_txt/Scene18/fog/info.txt', delimiter=' ')
scene18_fog_pose = pd.read_csv('./VKITTI_txt/Scene18/fog/pose.txt', delimiter=' ')

scene18_morning = os.listdir('./VKITTI/Scene18/morning/frames/rgb/Camera_1/')
scene18_morning_bbox = pd.read_csv('./VKITTI_txt/Scene18/morning/bbox.txt', delimiter=' ')
scene18_morning_info = pd.read_csv('./VKITTI_txt/Scene18/morning/info.txt', delimiter=' ')
scene18_morning_pose = pd.read_csv('./VKITTI_txt/Scene18/morning/pose.txt', delimiter=' ')

scene18_overcast = os.listdir('./VKITTI/Scene18/overcast/frames/rgb/Camera_1/')
scene18_overcast_bbox = pd.read_csv('./VKITTI_txt/Scene18/overcast/bbox.txt', delimiter=' ')
scene18_overcast_info = pd.read_csv('./VKITTI_txt/Scene18/overcast/info.txt', delimiter=' ')
scene18_overcast_pose = pd.read_csv('./VKITTI_txt/Scene18/overcast/pose.txt', delimiter=' ')

scene18_rain = os.listdir('./VKITTI/Scene18/rain/frames/rgb/Camera_1/')
scene18_rain_bbox = pd.read_csv('./VKITTI_txt/Scene18/rain/bbox.txt', delimiter=' ')
scene18_rain_info = pd.read_csv('./VKITTI_txt/Scene18/rain/info.txt', delimiter=' ')
scene18_rain_pose = pd.read_csv('./VKITTI_txt/Scene18/rain/pose.txt', delimiter=' ')

scene18_sunset = os.listdir('./VKITTI/Scene18/sunset/frames/rgb/Camera_1/')
scene18_sunset_bbox = pd.read_csv('./VKITTI_txt/Scene18/sunset/bbox.txt', delimiter=' ')
scene18_sunset_info = pd.read_csv('./VKITTI_txt/Scene18/sunset/info.txt', delimiter=' ')
scene18_sunset_pose = pd.read_csv('./VKITTI_txt/Scene18/sunset/pose.txt', delimiter=' ')

# scene 20
scene20_clone = os.listdir('./VKITTI/Scene20/clone/frames/rgb/Camera_1/')
scene20_clone_bbox = pd.read_csv('./VKITTI_txt/Scene20/clone/bbox.txt', delimiter=' ')
scene20_clone_info = pd.read_csv('./VKITTI_txt/Scene20/clone/info.txt', delimiter=' ')
scene20_clone_pose = pd.read_csv('./VKITTI_txt/Scene20/clone/pose.txt', delimiter=' ')

scene20_fog = os.listdir('./VKITTI/Scene20/fog/frames/rgb/Camera_1/')
scene20_fog_bbox = pd.read_csv('./VKITTI_txt/Scene20/fog/bbox.txt', delimiter=' ')
scene20_fog_info = pd.read_csv('./VKITTI_txt/Scene20/fog/info.txt', delimiter=' ')
scene20_fog_pose = pd.read_csv('./VKITTI_txt/Scene20/fog/pose.txt', delimiter=' ')

scene20_morning = os.listdir('./VKITTI/Scene20/morning/frames/rgb/Camera_1/')
scene20_morning_bbox = pd.read_csv('./VKITTI_txt/Scene20/morning/bbox.txt', delimiter=' ')
scene20_morning_info = pd.read_csv('./VKITTI_txt/Scene20/morning/info.txt', delimiter=' ')
scene20_morning_pose = pd.read_csv('./VKITTI_txt/Scene20/morning/pose.txt', delimiter=' ')

scene20_overcast = os.listdir('./VKITTI/Scene20/overcast/frames/rgb/Camera_1/')
scene20_overcast_bbox = pd.read_csv('./VKITTI_txt/Scene20/overcast/bbox.txt', delimiter=' ')
scene20_overcast_info = pd.read_csv('./VKITTI_txt/Scene20/overcast/info.txt', delimiter=' ')
scene20_overcast_pose = pd.read_csv('./VKITTI_txt/Scene20/overcast/pose.txt', delimiter=' ')

scene20_rain = os.listdir('./VKITTI/Scene20/rain/frames/rgb/Camera_1/')
scene20_rain_bbox = pd.read_csv('./VKITTI_txt/Scene20/rain/bbox.txt', delimiter=' ')
scene20_rain_info = pd.read_csv('./VKITTI_txt/Scene20/rain/info.txt', delimiter=' ')
scene20_rain_pose = pd.read_csv('./VKITTI_txt/Scene20/rain/pose.txt', delimiter=' ')

scene20_sunset = os.listdir('./VKITTI/Scene20/sunset/frames/rgb/Camera_1/')
scene20_sunset_bbox = pd.read_csv('./VKITTI_txt/Scene20/sunset/bbox.txt', delimiter=' ')
scene20_sunset_info = pd.read_csv('./VKITTI_txt/Scene20/sunset/info.txt', delimiter=' ')
scene20_sunset_pose = pd.read_csv('./VKITTI_txt/Scene20/sunset/pose.txt', delimiter=' ')

################################################################################################################################
# 3. Data Preprocessing and merge each Scene
'''
VKITTI2에 있는 사진과 txt파일을 알맞게 병합하여, 추후 DETR의 bounding box와 비교가 잘 될 수 있도록 format을 만든다.
'''

start = time.time() 

# Scene1
# scene1 Preprocessing
length = len(scene1_clone) // 6

scene1_vkitti_data = pd.DataFrame()

for i in tqdm(range(6)):
    print('scene1 중',(i+1),'번째 데이터 정제 중')
    # 데이터 feature 추출
    # 'clone'
    if i == 0:
    
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene1_clone_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene1_clone_bbox = scene1_clone_bbox[scene1_clone_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene1_clone[fr] for fr in scene1_clone_bbox['frame'].values]
        scene1_clone_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene1_clone_info[scene1_clone_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene1_clone_bbox['trackID'].values] 
        scene1_clone_bbox.drop('trackID', axis=1, inplace=True)

        scene1_clone_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene1_clone_pose = scene1_clone_pose[scene1_clone_pose['cameraID']==1][['angle','zloc']]

        scene1_clone_bbox.reset_index(inplace=True)
        scene1_clone_pose.reset_index(inplace=True)
        scene1_clone_bbox.drop('index', axis=1, inplace=True)
        scene1_clone_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene1_clone_data = pd.DataFrame({'filename':filename})
        scene1_clone_data['class'] = label_list
        scene1_clone_data = pd.concat([scene1_clone_data, scene1_clone_bbox], axis=1)
        scene1_clone_data = pd.concat([scene1_clone_data, scene1_clone_pose], axis=1)
        scene1_clone_data['weather'] = 'clone'
        scene1_clone_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene1_clone[length*i:length*(i+1)]
        mask = [index for index in range(len(scene1_clone_data)) if scene1_clone_data['filename'].values[index] in set_filename]
        scene1_clone_data = scene1_clone_data.iloc[mask]

        # 데이터 최종 병합
        scene1_vkitti_data = pd.concat([scene1_vkitti_data, scene1_clone_data], axis=0)
        
    # 'fog'
    elif i == 1:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene1_fog_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene1_fog_bbox = scene1_fog_bbox[scene1_fog_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene1_fog[fr] for fr in scene1_fog_bbox['frame'].values]
        scene1_fog_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene1_fog_info[scene1_fog_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene1_fog_bbox['trackID'].values] 
        scene1_fog_bbox.drop('trackID', axis=1, inplace=True)

        scene1_fog_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene1_fog_pose = scene1_fog_pose[scene1_fog_pose['cameraID']==1][['angle','zloc']]

        scene1_fog_bbox.reset_index(inplace=True)
        scene1_fog_pose.reset_index(inplace=True)
        scene1_fog_bbox.drop('index', axis=1, inplace=True)
        scene1_fog_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene1_fog_data = pd.DataFrame({'filename':filename})
        scene1_fog_data['class'] = label_list
        scene1_fog_data = pd.concat([scene1_fog_data, scene1_fog_bbox], axis=1)
        scene1_fog_data = pd.concat([scene1_fog_data, scene1_fog_pose], axis=1)
        scene1_fog_data['weather'] = 'fog'
        scene1_fog_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene1_fog[length*i:length*(i+1)]
        mask = [index for index in range(len(scene1_fog_data)) if scene1_fog_data['filename'].values[index] in set_filename]
        scene1_fog_data = scene1_fog_data.iloc[mask]

        # 데이터 최종 병합
        scene1_vkitti_data = pd.concat([scene1_vkitti_data, scene1_fog_data], axis=0)
        
    # 'morning'
    elif i == 2:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene1_morning_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene1_morning_bbox = scene1_morning_bbox[scene1_morning_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene1_morning[fr] for fr in scene1_morning_bbox['frame'].values]
        scene1_morning_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene1_morning_info[scene1_morning_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene1_morning_bbox['trackID'].values] 
        scene1_morning_bbox.drop('trackID', axis=1, inplace=True)

        scene1_morning_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene1_morning_pose = scene1_morning_pose[scene1_morning_pose['cameraID']==1][['angle','zloc']]

        scene1_morning_bbox.reset_index(inplace=True)
        scene1_morning_pose.reset_index(inplace=True)
        scene1_morning_bbox.drop('index', axis=1, inplace=True)
        scene1_morning_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene1_morning_data = pd.DataFrame({'filename':filename})
        scene1_morning_data['class'] = label_list
        scene1_morning_data = pd.concat([scene1_morning_data, scene1_morning_bbox], axis=1)
        scene1_morning_data = pd.concat([scene1_morning_data, scene1_morning_pose], axis=1)
        scene1_morning_data['weather'] = 'morning'
        scene1_morning_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene1_morning[length*i:length*(i+1)]
        mask = [index for index in range(len(scene1_morning_data)) if scene1_morning_data['filename'].values[index] in set_filename]
        scene1_morning_data = scene1_morning_data.iloc[mask]

        # 데이터 최종 병합
        scene1_vkitti_data = pd.concat([scene1_vkitti_data, scene1_morning_data], axis=0)
        
    # 'overcast'
    elif i == 3:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene1_overcast_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene1_overcast_bbox = scene1_overcast_bbox[scene1_overcast_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene1_overcast[fr] for fr in scene1_overcast_bbox['frame'].values]
        scene1_overcast_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene1_overcast_info[scene1_overcast_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene1_overcast_bbox['trackID'].values] 
        scene1_overcast_bbox.drop('trackID', axis=1, inplace=True)

        scene1_overcast_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene1_overcast_pose = scene1_overcast_pose[scene1_overcast_pose['cameraID']==1][['angle','zloc']]

        scene1_overcast_bbox.reset_index(inplace=True)
        scene1_overcast_pose.reset_index(inplace=True)
        scene1_overcast_bbox.drop('index', axis=1, inplace=True)
        scene1_overcast_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene1_overcast_data = pd.DataFrame({'filename':filename})
        scene1_overcast_data['class'] = label_list
        scene1_overcast_data = pd.concat([scene1_overcast_data, scene1_overcast_bbox], axis=1)
        scene1_overcast_data = pd.concat([scene1_overcast_data, scene1_overcast_pose], axis=1)
        scene1_overcast_data['weather'] = 'overcast'
        scene1_overcast_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene1_overcast[length*i:length*(i+1)]
        mask = [index for index in range(len(scene1_overcast_data)) if scene1_overcast_data['filename'].values[index] in set_filename]
        scene1_overcast_data = scene1_overcast_data.iloc[mask]

        # 데이터 최종 병합
        scene1_vkitti_data = pd.concat([scene1_vkitti_data, scene1_overcast_data], axis=0)
        
    # 'rain'
    elif i == 4:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene1_rain_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene1_rain_bbox = scene1_rain_bbox[scene1_rain_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene1_rain[fr] for fr in scene1_rain_bbox['frame'].values]
        scene1_rain_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene1_rain_info[scene1_rain_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene1_rain_bbox['trackID'].values] 
        scene1_rain_bbox.drop('trackID', axis=1, inplace=True)

        scene1_rain_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene1_rain_pose = scene1_rain_pose[scene1_rain_pose['cameraID']==1][['angle','zloc']]

        scene1_rain_bbox.reset_index(inplace=True)
        scene1_rain_pose.reset_index(inplace=True)
        scene1_rain_bbox.drop('index', axis=1, inplace=True)
        scene1_rain_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene1_rain_data = pd.DataFrame({'filename':filename})
        scene1_rain_data['class'] = label_list
        scene1_rain_data = pd.concat([scene1_rain_data, scene1_rain_bbox], axis=1)
        scene1_rain_data = pd.concat([scene1_rain_data, scene1_rain_pose], axis=1)
        scene1_rain_data['weather'] = 'rain'
        scene1_rain_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene1_rain[length*i:length*(i+1)]
        mask = [index for index in range(len(scene1_rain_data)) if scene1_rain_data['filename'].values[index] in set_filename]
        scene1_rain_data = scene1_rain_data.iloc[mask]

        # 데이터 최종 병합
        scene1_vkitti_data = pd.concat([scene1_vkitti_data, scene1_rain_data], axis=0)
        
    # 'sunset'
    elif i == 5:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene1_sunset_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene1_sunset_bbox = scene1_sunset_bbox[scene1_sunset_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene1_sunset[fr] for fr in scene1_sunset_bbox['frame'].values]
        scene1_sunset_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene1_sunset_info[scene1_sunset_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene1_sunset_bbox['trackID'].values] 
        scene1_sunset_bbox.drop('trackID', axis=1, inplace=True)

        scene1_sunset_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene1_sunset_pose = scene1_sunset_pose[scene1_sunset_pose['cameraID']==1][['angle','zloc']]

        scene1_sunset_bbox.reset_index(inplace=True)
        scene1_sunset_pose.reset_index(inplace=True)
        scene1_sunset_bbox.drop('index', axis=1, inplace=True)
        scene1_sunset_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene1_sunset_data = pd.DataFrame({'filename':filename})
        scene1_sunset_data['class'] = label_list
        scene1_sunset_data = pd.concat([scene1_sunset_data, scene1_sunset_bbox], axis=1)
        scene1_sunset_data = pd.concat([scene1_sunset_data, scene1_sunset_pose], axis=1)
        scene1_sunset_data['weather'] = 'sunset'
        scene1_sunset_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene1_sunset[length*i:length*(i+1)]
        mask = [index for index in range(len(scene1_sunset_data)) if scene1_sunset_data['filename'].values[index] in set_filename]
        scene1_sunset_data = scene1_sunset_data.iloc[mask]

        # 데이터 최종 병합
        scene1_vkitti_data = pd.concat([scene1_vkitti_data, scene1_sunset_data], axis=0)

scene1_vkitti_data['filename'] = ['scene1_'+name for name in scene1_vkitti_data['filename']]

scene1_vkitti_data.isnull().sum(axis=0)
#scene1_vkitti_data.to_csv('./scene1_vkitti_data.csv', mode='a', index=False)

print('scene1 end')


#########################################################################################################################
# Scene2
# scene2 Preprocessing
length = len(scene2_clone) // 6

scene2_vkitti_data = pd.DataFrame()

for i in tqdm(range(6)):
    print('scene2 중',(i+1),'번째 데이터 정제 중')
    # 데이터 feature 추출
    # 'clone'
    if i == 0:
    
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene2_clone_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene2_clone_bbox = scene2_clone_bbox[scene2_clone_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene2_clone[fr] for fr in scene2_clone_bbox['frame'].values]
        scene2_clone_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene2_clone_info[scene2_clone_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene2_clone_bbox['trackID'].values] 
        scene2_clone_bbox.drop('trackID', axis=1, inplace=True)

        scene2_clone_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene2_clone_pose = scene2_clone_pose[scene2_clone_pose['cameraID']==1][['angle','zloc']]

        scene2_clone_bbox.reset_index(inplace=True)
        scene2_clone_pose.reset_index(inplace=True)
        scene2_clone_bbox.drop('index', axis=1, inplace=True)
        scene2_clone_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene2_clone_data = pd.DataFrame({'filename':filename})
        scene2_clone_data['class'] = label_list
        scene2_clone_data = pd.concat([scene2_clone_data, scene2_clone_bbox], axis=1)
        scene2_clone_data = pd.concat([scene2_clone_data, scene2_clone_pose], axis=1)
        scene2_clone_data['weather'] = 'clone'
        scene2_clone_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene2_clone[length*(i):length*(i+1)]
        mask = [index for index in range(len(scene2_clone_data)) if scene2_clone_data['filename'].values[index] in set_filename]
        scene2_clone_data = scene2_clone_data.iloc[mask]

        # 데이터 최종 병합
        scene2_vkitti_data = pd.concat([scene2_vkitti_data, scene2_clone_data], axis=0)
        
    # 'fog'
    elif i == 1:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene2_fog_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene2_fog_bbox = scene2_fog_bbox[scene2_fog_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene2_fog[fr] for fr in scene2_fog_bbox['frame'].values]
        scene2_fog_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene2_fog_info[scene2_fog_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene2_fog_bbox['trackID'].values] 
        scene2_fog_bbox.drop('trackID', axis=1, inplace=True)

        scene2_fog_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene2_fog_pose = scene2_fog_pose[scene2_fog_pose['cameraID']==1][['angle','zloc']]

        scene2_fog_bbox.reset_index(inplace=True)
        scene2_fog_pose.reset_index(inplace=True)
        scene2_fog_bbox.drop('index', axis=1, inplace=True)
        scene2_fog_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene2_fog_data = pd.DataFrame({'filename':filename})
        scene2_fog_data['class'] = label_list
        scene2_fog_data = pd.concat([scene2_fog_data, scene2_fog_bbox], axis=1)
        scene2_fog_data = pd.concat([scene2_fog_data, scene2_fog_pose], axis=1)
        scene2_fog_data['weather'] = 'fog'
        scene2_fog_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene2_fog[length*i:length*(i+1)]
        mask = [index for index in range(len(scene2_fog_data)) if scene2_fog_data['filename'].values[index] in set_filename]
        scene2_fog_data = scene2_fog_data.iloc[mask]
        
        # 데이터 최종 병합
        scene2_vkitti_data = pd.concat([scene2_vkitti_data, scene2_fog_data], axis=0)
        
    # 'morning'
    elif i == 2:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene2_morning_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene2_morning_bbox = scene2_morning_bbox[scene2_morning_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene2_morning[fr] for fr in scene2_morning_bbox['frame'].values]
        scene2_morning_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene2_morning_info[scene2_morning_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene2_morning_bbox['trackID'].values] 
        scene2_morning_bbox.drop('trackID', axis=1, inplace=True)

        scene2_morning_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene2_morning_pose = scene2_morning_pose[scene2_morning_pose['cameraID']==1][['angle','zloc']]

        scene2_morning_bbox.reset_index(inplace=True)
        scene2_morning_pose.reset_index(inplace=True)
        scene2_morning_bbox.drop('index', axis=1, inplace=True)
        scene2_morning_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene2_morning_data = pd.DataFrame({'filename':filename})
        scene2_morning_data['class'] = label_list
        scene2_morning_data = pd.concat([scene2_morning_data, scene2_morning_bbox], axis=1)
        scene2_morning_data = pd.concat([scene2_morning_data, scene2_morning_pose], axis=1)
        scene2_morning_data['weather'] = 'morning'
        scene2_morning_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene2_morning[length*i:length*(i+1)]
        mask = [index for index in range(len(scene2_morning_data)) if scene2_morning_data['filename'].values[index] in set_filename]
        scene2_morning_data = scene2_morning_data.iloc[mask]

        # 데이터 최종 병합
        scene2_vkitti_data = pd.concat([scene2_vkitti_data, scene2_morning_data], axis=0)
        
    # 'overcast'
    elif i == 3:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene2_overcast_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene2_overcast_bbox = scene2_overcast_bbox[scene2_overcast_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene2_overcast[fr] for fr in scene2_overcast_bbox['frame'].values]
        scene2_overcast_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene2_overcast_info[scene2_overcast_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene2_overcast_bbox['trackID'].values] 
        scene2_overcast_bbox.drop('trackID', axis=1, inplace=True)

        scene2_overcast_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene2_overcast_pose = scene2_overcast_pose[scene2_overcast_pose['cameraID']==1][['angle','zloc']]

        scene2_overcast_bbox.reset_index(inplace=True)
        scene2_overcast_pose.reset_index(inplace=True)
        scene2_overcast_bbox.drop('index', axis=1, inplace=True)
        scene2_overcast_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene2_overcast_data = pd.DataFrame({'filename':filename})
        scene2_overcast_data['class'] = label_list
        scene2_overcast_data = pd.concat([scene2_overcast_data, scene2_overcast_bbox], axis=1)
        scene2_overcast_data = pd.concat([scene2_overcast_data, scene2_overcast_pose], axis=1)
        scene2_overcast_data['weather'] = 'overcast'
        scene2_overcast_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene2_overcast[length*i:length*(i+1)]
        mask = [index for index in range(len(scene2_overcast_data)) if scene2_overcast_data['filename'].values[index] in set_filename]
        scene2_overcast_data = scene2_overcast_data.iloc[mask]

        # 데이터 최종 병합
        scene2_vkitti_data = pd.concat([scene2_vkitti_data, scene2_overcast_data], axis=0)
        
    # 'rain'
    elif i == 4:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene2_rain_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene2_rain_bbox = scene2_rain_bbox[scene2_rain_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene2_rain[fr] for fr in scene2_rain_bbox['frame'].values]
        scene2_rain_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene2_rain_info[scene2_rain_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene2_rain_bbox['trackID'].values] 
        scene2_rain_bbox.drop('trackID', axis=1, inplace=True)

        scene2_rain_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene2_rain_pose = scene2_rain_pose[scene2_rain_pose['cameraID']==1][['angle','zloc']]

        scene2_rain_bbox.reset_index(inplace=True)
        scene2_rain_pose.reset_index(inplace=True)
        scene2_rain_bbox.drop('index', axis=1, inplace=True)
        scene2_rain_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene2_rain_data = pd.DataFrame({'filename':filename})
        scene2_rain_data['class'] = label_list
        scene2_rain_data = pd.concat([scene2_rain_data, scene2_rain_bbox], axis=1)
        scene2_rain_data = pd.concat([scene2_rain_data, scene2_rain_pose], axis=1)
        scene2_rain_data['weather'] = 'rain'
        scene2_rain_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene2_rain[length*i:length*(i+1)]
        mask = [index for index in range(len(scene2_rain_data)) if scene2_rain_data['filename'].values[index] in set_filename]
        scene2_rain_data = scene2_rain_data.iloc[mask]

        # 데이터 최종 병합
        scene2_vkitti_data = pd.concat([scene2_vkitti_data, scene2_rain_data], axis=0)
        
    # 'sunset'
    elif i == 5:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene2_sunset_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene2_sunset_bbox = scene2_sunset_bbox[scene2_sunset_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene2_sunset[fr] for fr in scene2_sunset_bbox['frame'].values]
        scene2_sunset_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene2_sunset_info[scene2_sunset_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene2_sunset_bbox['trackID'].values] 
        scene2_sunset_bbox.drop('trackID', axis=1, inplace=True)

        scene2_sunset_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene2_sunset_pose = scene2_sunset_pose[scene2_sunset_pose['cameraID']==1][['angle','zloc']]

        scene2_sunset_bbox.reset_index(inplace=True)
        scene2_sunset_pose.reset_index(inplace=True)
        scene2_sunset_bbox.drop('index', axis=1, inplace=True)
        scene2_sunset_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene2_sunset_data = pd.DataFrame({'filename':filename})
        scene2_sunset_data['class'] = label_list
        scene2_sunset_data = pd.concat([scene2_sunset_data, scene2_sunset_bbox], axis=1)
        scene2_sunset_data = pd.concat([scene2_sunset_data, scene2_sunset_pose], axis=1)
        scene2_sunset_data['weather'] = 'sunset'
        scene2_sunset_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene2_sunset[length*i:length*(i+1)]
        mask = [index for index in range(len(scene2_sunset_data)) if scene2_sunset_data['filename'].values[index] in set_filename]
        scene2_sunset_data = scene2_sunset_data.iloc[mask]

        # 데이터 최종 병합
        scene2_vkitti_data = pd.concat([scene2_vkitti_data, scene2_sunset_data], axis=0)
        
scene2_vkitti_data['filename'] = ['scene2_'+name for name in scene2_vkitti_data['filename']]

scene2_vkitti_data.isnull().sum(axis=0)
#scene2_vkitti_data.to_csv('./scene2_vkitti_data.csv', mode='a', index=False)

print('scene2 end')

#########################################################################################################################
# Scene6
# scene6 Preprocessing
length = len(scene6_clone) // 6

scene6_vkitti_data = pd.DataFrame()

for i in tqdm(range(6)):
    print('scene6 중',(i+1),'번째 데이터 정제 중')
    # 데이터 feature 추출
    # 'clone'
    if i == 0:
    
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene6_clone_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene6_clone_bbox = scene6_clone_bbox[scene6_clone_bbox['cameraID']==6][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene6_clone[fr] for fr in scene6_clone_bbox['frame'].values]
        scene6_clone_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene6_clone_info[scene6_clone_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene6_clone_bbox['trackID'].values] 
        scene6_clone_bbox.drop('trackID', axis=1, inplace=True)

        scene6_clone_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene6_clone_pose = scene6_clone_pose[scene6_clone_pose['cameraID']==1][['angle','zloc']]

        scene6_clone_bbox.reset_index(inplace=True)
        scene6_clone_pose.reset_index(inplace=True)
        scene6_clone_bbox.drop('index', axis=1, inplace=True)
        scene6_clone_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene6_clone_data = pd.DataFrame({'filename':filename})
        scene6_clone_data['class'] = label_list
        scene6_clone_data = pd.concat([scene6_clone_data, scene6_clone_bbox], axis=1)
        scene6_clone_data = pd.concat([scene6_clone_data, scene6_clone_pose], axis=1)
        scene6_clone_data['weather'] = 'clone'
        scene6_clone_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene6_clone[length*i:length*(i+1)]
        mask = [index for index in range(len(scene6_clone_data)) if scene6_clone_data['filename'].values[index] in set_filename]
        scene6_clone_data = scene6_clone_data.iloc[mask]

        # 데이터 최종 병합
        scene6_vkitti_data = pd.concat([scene6_vkitti_data, scene6_clone_data], axis=0)
        
    # 'fog'
    elif i == 1:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene6_fog_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene6_fog_bbox = scene6_fog_bbox[scene6_fog_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene6_fog[fr] for fr in scene6_fog_bbox['frame'].values]
        scene6_fog_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene6_fog_info[scene6_fog_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene6_fog_bbox['trackID'].values] 
        scene6_fog_bbox.drop('trackID', axis=1, inplace=True)

        scene6_fog_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene6_fog_pose = scene6_fog_pose[scene6_fog_pose['cameraID']==1][['angle','zloc']]

        scene6_fog_bbox.reset_index(inplace=True)
        scene6_fog_pose.reset_index(inplace=True)
        scene6_fog_bbox.drop('index', axis=1, inplace=True)
        scene6_fog_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene6_fog_data = pd.DataFrame({'filename':filename})
        scene6_fog_data['class'] = label_list
        scene6_fog_data = pd.concat([scene6_fog_data, scene6_fog_bbox], axis=1)
        scene6_fog_data = pd.concat([scene6_fog_data, scene6_fog_pose], axis=1)
        scene6_fog_data['weather'] = 'fog'
        scene6_fog_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene6_fog[length*i:length*(i+1)]
        mask = [index for index in range(len(scene6_fog_data)) if scene6_fog_data['filename'].values[index] in set_filename]
        scene6_fog_data = scene6_fog_data.iloc[mask]

        # 데이터 최종 병합
        scene6_vkitti_data = pd.concat([scene6_vkitti_data, scene6_fog_data], axis=0)
        
    # 'morning'
    elif i == 2:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene6_morning_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene6_morning_bbox = scene6_morning_bbox[scene6_morning_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene6_morning[fr] for fr in scene6_morning_bbox['frame'].values]
        scene6_morning_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene6_morning_info[scene6_morning_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene6_morning_bbox['trackID'].values] 
        scene6_morning_bbox.drop('trackID', axis=1, inplace=True)

        scene6_morning_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene6_morning_pose = scene6_morning_pose[scene6_morning_pose['cameraID']==1][['angle','zloc']]

        scene6_morning_bbox.reset_index(inplace=True)
        scene6_morning_pose.reset_index(inplace=True)
        scene6_morning_bbox.drop('index', axis=1, inplace=True)
        scene6_morning_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene6_morning_data = pd.DataFrame({'filename':filename})
        scene6_morning_data['class'] = label_list
        scene6_morning_data = pd.concat([scene6_morning_data, scene6_morning_bbox], axis=1)
        scene6_morning_data = pd.concat([scene6_morning_data, scene6_morning_pose], axis=1)
        scene6_morning_data['weather'] = 'morning'
        scene6_morning_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene6_morning[length*i:length*(i+1)]
        mask = [index for index in range(len(scene6_morning_data)) if scene6_morning_data['filename'].values[index] in set_filename]
        scene6_morning_data = scene6_morning_data.iloc[mask]

        # 데이터 최종 병합
        scene6_vkitti_data = pd.concat([scene6_vkitti_data, scene6_morning_data], axis=0)
        
    # 'overcast'
    elif i == 3:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene6_overcast_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene6_overcast_bbox = scene6_overcast_bbox[scene6_overcast_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene6_overcast[fr] for fr in scene6_overcast_bbox['frame'].values]
        scene6_overcast_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene6_overcast_info[scene6_overcast_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene6_overcast_bbox['trackID'].values] 
        scene6_overcast_bbox.drop('trackID', axis=1, inplace=True)

        scene6_overcast_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene6_overcast_pose = scene6_overcast_pose[scene6_overcast_pose['cameraID']==1][['angle','zloc']]

        scene6_overcast_bbox.reset_index(inplace=True)
        scene6_overcast_pose.reset_index(inplace=True)
        scene6_overcast_bbox.drop('index', axis=1, inplace=True)
        scene6_overcast_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene6_overcast_data = pd.DataFrame({'filename':filename})
        scene6_overcast_data['class'] = label_list
        scene6_overcast_data = pd.concat([scene6_overcast_data, scene6_overcast_bbox], axis=1)
        scene6_overcast_data = pd.concat([scene6_overcast_data, scene6_overcast_pose], axis=1)
        scene6_overcast_data['weather'] = 'overcast'
        scene6_overcast_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene6_overcast[length*i:length*(i+1)]
        mask = [index for index in range(len(scene6_overcast_data)) if scene6_overcast_data['filename'].values[index] in set_filename]
        scene6_overcast_data = scene6_overcast_data.iloc[mask]

        # 데이터 최종 병합
        scene6_vkitti_data = pd.concat([scene6_vkitti_data, scene6_overcast_data], axis=0)
        
    # 'rain'
    elif i == 4:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene6_rain_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene6_rain_bbox = scene6_rain_bbox[scene6_rain_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene6_rain[fr] for fr in scene6_rain_bbox['frame'].values]
        scene6_rain_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene6_rain_info[scene6_rain_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene6_rain_bbox['trackID'].values] 
        scene6_rain_bbox.drop('trackID', axis=1, inplace=True)

        scene6_rain_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene6_rain_pose = scene6_rain_pose[scene6_rain_pose['cameraID']==1][['angle','zloc']]

        scene6_rain_bbox.reset_index(inplace=True)
        scene6_rain_pose.reset_index(inplace=True)
        scene6_rain_bbox.drop('index', axis=1, inplace=True)
        scene6_rain_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene6_rain_data = pd.DataFrame({'filename':filename})
        scene6_rain_data['class'] = label_list
        scene6_rain_data = pd.concat([scene6_rain_data, scene6_rain_bbox], axis=1)
        scene6_rain_data = pd.concat([scene6_rain_data, scene6_rain_pose], axis=1)
        scene6_rain_data['weather'] = 'rain'
        scene6_rain_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene6_rain[length*i:length*(i+1)]
        mask = [index for index in range(len(scene6_rain_data)) if scene6_rain_data['filename'].values[index] in set_filename]
        scene6_rain_data = scene6_rain_data.iloc[mask]

        # 데이터 최종 병합
        scene6_vkitti_data = pd.concat([scene6_vkitti_data, scene6_rain_data], axis=0)
        
    # 'sunset'
    elif i == 5:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene6_sunset_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene6_sunset_bbox = scene6_sunset_bbox[scene6_sunset_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene6_sunset[fr] for fr in scene6_sunset_bbox['frame'].values]
        scene6_sunset_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene6_sunset_info[scene6_sunset_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene6_sunset_bbox['trackID'].values] 
        scene6_sunset_bbox.drop('trackID', axis=1, inplace=True)

        scene6_sunset_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene6_sunset_pose = scene6_sunset_pose[scene6_sunset_pose['cameraID']==1][['angle','zloc']]

        scene6_sunset_bbox.reset_index(inplace=True)
        scene6_sunset_pose.reset_index(inplace=True)
        scene6_sunset_bbox.drop('index', axis=1, inplace=True)
        scene6_sunset_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene6_sunset_data = pd.DataFrame({'filename':filename})
        scene6_sunset_data['class'] = label_list
        scene6_sunset_data = pd.concat([scene6_sunset_data, scene6_sunset_bbox], axis=1)
        scene6_sunset_data = pd.concat([scene6_sunset_data, scene6_sunset_pose], axis=1)
        scene6_sunset_data['weather'] = 'sunset'
        scene6_sunset_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene6_sunset[length*i:length*(i+1)]
        mask = [index for index in range(len(scene6_sunset_data)) if scene6_sunset_data['filename'].values[index] in set_filename]
        scene6_sunset_data = scene6_sunset_data.iloc[mask]

        # 데이터 최종 병합
        scene6_vkitti_data = pd.concat([scene6_vkitti_data, scene6_sunset_data], axis=0)
        
scene6_vkitti_data['filename'] = ['scene6_'+name for name in scene6_vkitti_data['filename']]

scene6_vkitti_data.isnull().sum(axis=0)
#scene6_vkitti_data.to_csv('./scene6_vkitti_data.csv', mode='a', index=False)

print('scene6 end')

#########################################################################################################################
# Scene18
# scene18 Preprocessing
length = len(scene18_clone) // 6

scene18_vkitti_data = pd.DataFrame()

for i in tqdm(range(6)):
    print('scene18 중',(i+1),'번째 데이터 정제 중')
    # 데이터 feature 추출
    # 'clone'
    if i == 0:
    
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene18_clone_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene18_clone_bbox = scene18_clone_bbox[scene18_clone_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene18_clone[fr] for fr in scene18_clone_bbox['frame'].values]
        scene18_clone_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene18_clone_info[scene18_clone_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene18_clone_bbox['trackID'].values] 
        scene18_clone_bbox.drop('trackID', axis=1, inplace=True)

        scene18_clone_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene18_clone_pose = scene18_clone_pose[scene18_clone_pose['cameraID']==1][['angle','zloc']]

        scene18_clone_bbox.reset_index(inplace=True)
        scene18_clone_pose.reset_index(inplace=True)
        scene18_clone_bbox.drop('index', axis=1, inplace=True)
        scene18_clone_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene18_clone_data = pd.DataFrame({'filename':filename})
        scene18_clone_data['class'] = label_list
        scene18_clone_data = pd.concat([scene18_clone_data, scene18_clone_bbox], axis=1)
        scene18_clone_data = pd.concat([scene18_clone_data, scene18_clone_pose], axis=1)
        scene18_clone_data['weather'] = 'clone'
        scene18_clone_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene18_clone[length*i:length*(i+1)]
        mask = [index for index in range(len(scene18_clone_data)) if scene18_clone_data['filename'].values[index] in set_filename]
        scene18_clone_data = scene18_clone_data.iloc[mask]

        # 데이터 최종 병합
        scene18_vkitti_data = pd.concat([scene18_vkitti_data, scene18_clone_data], axis=0)
        
    # 'fog'
    elif i == 1:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene18_fog_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene18_fog_bbox = scene18_fog_bbox[scene18_fog_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene18_fog[fr] for fr in scene18_fog_bbox['frame'].values]
        scene18_fog_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene18_fog_info[scene18_fog_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene18_fog_bbox['trackID'].values] 
        scene18_fog_bbox.drop('trackID', axis=1, inplace=True)

        scene18_fog_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene18_fog_pose = scene18_fog_pose[scene18_fog_pose['cameraID']==1][['angle','zloc']]

        scene18_fog_bbox.reset_index(inplace=True)
        scene18_fog_pose.reset_index(inplace=True)
        scene18_fog_bbox.drop('index', axis=1, inplace=True)
        scene18_fog_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene18_fog_data = pd.DataFrame({'filename':filename})
        scene18_fog_data['class'] = label_list
        scene18_fog_data = pd.concat([scene18_fog_data, scene18_fog_bbox], axis=1)
        scene18_fog_data = pd.concat([scene18_fog_data, scene18_fog_pose], axis=1)
        scene18_fog_data['weather'] = 'fog'
        scene18_fog_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene18_fog[length*i:length*(i+1)]
        mask = [index for index in range(len(scene18_fog_data)) if scene18_fog_data['filename'].values[index] in set_filename]
        scene18_fog_data = scene18_fog_data.iloc[mask]

        # 데이터 최종 병합
        scene18_vkitti_data = pd.concat([scene18_vkitti_data, scene18_fog_data], axis=0)
        
    # 'morning'
    elif i == 2:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene18_morning_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene18_morning_bbox = scene18_morning_bbox[scene18_morning_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene18_morning[fr] for fr in scene18_morning_bbox['frame'].values]
        scene18_morning_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene18_morning_info[scene18_morning_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene18_morning_bbox['trackID'].values] 
        scene18_morning_bbox.drop('trackID', axis=1, inplace=True)

        scene18_morning_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene18_morning_pose = scene18_morning_pose[scene18_morning_pose['cameraID']==1][['angle','zloc']]

        scene18_morning_bbox.reset_index(inplace=True)
        scene18_morning_pose.reset_index(inplace=True)
        scene18_morning_bbox.drop('index', axis=1, inplace=True)
        scene18_morning_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene18_morning_data = pd.DataFrame({'filename':filename})
        scene18_morning_data['class'] = label_list
        scene18_morning_data = pd.concat([scene18_morning_data, scene18_morning_bbox], axis=1)
        scene18_morning_data = pd.concat([scene18_morning_data, scene18_morning_pose], axis=1)
        scene18_morning_data['weather'] = 'morning'
        scene18_morning_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene18_morning[length*i:length*(i+1)]
        mask = [index for index in range(len(scene18_morning_data)) if scene18_morning_data['filename'].values[index] in set_filename]
        scene18_morning_data = scene18_morning_data.iloc[mask]

        # 데이터 최종 병합
        scene18_vkitti_data = pd.concat([scene18_vkitti_data, scene18_morning_data], axis=0)
        
    # 'overcast'
    elif i == 3:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene18_overcast_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene18_overcast_bbox = scene18_overcast_bbox[scene18_overcast_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene18_overcast[fr] for fr in scene18_overcast_bbox['frame'].values]
        scene18_overcast_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene18_overcast_info[scene18_overcast_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene18_overcast_bbox['trackID'].values] 
        scene18_overcast_bbox.drop('trackID', axis=1, inplace=True)

        scene18_overcast_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene18_overcast_pose = scene18_overcast_pose[scene18_overcast_pose['cameraID']==1][['angle','zloc']]

        scene18_overcast_bbox.reset_index(inplace=True)
        scene18_overcast_pose.reset_index(inplace=True)
        scene18_overcast_bbox.drop('index', axis=1, inplace=True)
        scene18_overcast_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene18_overcast_data = pd.DataFrame({'filename':filename})
        scene18_overcast_data['class'] = label_list
        scene18_overcast_data = pd.concat([scene18_overcast_data, scene18_overcast_bbox], axis=1)
        scene18_overcast_data = pd.concat([scene18_overcast_data, scene18_overcast_pose], axis=1)
        scene18_overcast_data['weather'] = 'overcast'
        scene18_overcast_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene18_overcast[length*i:length*(i+1)]
        mask = [index for index in range(len(scene18_overcast_data)) if scene18_overcast_data['filename'].values[index] in set_filename]
        scene18_overcast_data = scene18_overcast_data.iloc[mask]

        # 데이터 최종 병합
        scene18_vkitti_data = pd.concat([scene18_vkitti_data, scene18_overcast_data], axis=0)
        
    # 'rain'
    elif i == 4:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene18_rain_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene18_rain_bbox = scene18_rain_bbox[scene18_rain_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene18_rain[fr] for fr in scene18_rain_bbox['frame'].values]
        scene18_rain_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene18_rain_info[scene18_rain_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene18_rain_bbox['trackID'].values] 
        scene18_rain_bbox.drop('trackID', axis=1, inplace=True)

        scene18_rain_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene18_rain_pose = scene18_rain_pose[scene18_rain_pose['cameraID']==1][['angle','zloc']]

        scene18_rain_bbox.reset_index(inplace=True)
        scene18_rain_pose.reset_index(inplace=True)
        scene18_rain_bbox.drop('index', axis=1, inplace=True)
        scene18_rain_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene18_rain_data = pd.DataFrame({'filename':filename})
        scene18_rain_data['class'] = label_list
        scene18_rain_data = pd.concat([scene18_rain_data, scene18_rain_bbox], axis=1)
        scene18_rain_data = pd.concat([scene18_rain_data, scene18_rain_pose], axis=1)
        scene18_rain_data['weather'] = 'rain'
        scene18_rain_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene18_rain[length*i:length*(i+1)]
        mask = [index for index in range(len(scene18_rain_data)) if scene18_rain_data['filename'].values[index] in set_filename]
        scene18_rain_data = scene18_rain_data.iloc[mask]

        # 데이터 최종 병합
        scene18_vkitti_data = pd.concat([scene18_vkitti_data, scene18_rain_data], axis=0)
        
    # 'sunset'
    elif i == 5:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene18_sunset_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene18_sunset_bbox = scene18_sunset_bbox[scene18_sunset_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene18_sunset[fr] for fr in scene18_sunset_bbox['frame'].values]
        scene18_sunset_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene18_sunset_info[scene18_sunset_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene18_sunset_bbox['trackID'].values] 
        scene18_sunset_bbox.drop('trackID', axis=1, inplace=True)

        scene18_sunset_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene18_sunset_pose = scene18_sunset_pose[scene18_sunset_pose['cameraID']==1][['angle','zloc']]

        scene18_sunset_bbox.reset_index(inplace=True)
        scene18_sunset_pose.reset_index(inplace=True)
        scene18_sunset_bbox.drop('index', axis=1, inplace=True)
        scene18_sunset_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene18_sunset_data = pd.DataFrame({'filename':filename})
        scene18_sunset_data['class'] = label_list
        scene18_sunset_data = pd.concat([scene18_sunset_data, scene18_sunset_bbox], axis=1)
        scene18_sunset_data = pd.concat([scene18_sunset_data, scene18_sunset_pose], axis=1)
        scene18_sunset_data['weather'] = 'sunset'
        scene18_sunset_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene18_sunset[length*i:length*(i+1)]
        mask = [index for index in range(len(scene18_sunset_data)) if scene18_sunset_data['filename'].values[index] in set_filename]
        scene18_sunset_data = scene18_sunset_data.iloc[mask]

        # 데이터 최종 병합
        scene18_vkitti_data = pd.concat([scene18_vkitti_data, scene18_sunset_data], axis=0)
        
scene18_vkitti_data['filename'] = ['scene18_'+name for name in scene18_vkitti_data['filename']]

scene18_vkitti_data.isnull().sum(axis=0)
#scene18_vkitti_data.to_csv('./scene18_vkitti_data.csv', mode='a', index=False)

print('scene18 end')

#########################################################################################################################
# Scene20
# scene20 Preprocessing
length = len(scene20_clone) // 6

scene20_vkitti_data = pd.DataFrame()

for i in tqdm(range(6)):
    print('scene20 중',(i+1),'번째 데이터 정제 중')
    # 데이터 feature 추출
    # 'clone'
    if i == 0:
    
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene20_clone_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene20_clone_bbox = scene20_clone_bbox[scene20_clone_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene20_clone[fr] for fr in scene20_clone_bbox['frame'].values]
        scene20_clone_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene20_clone_info[scene20_clone_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene20_clone_bbox['trackID'].values] 
        scene20_clone_bbox.drop('trackID', axis=1, inplace=True)

        scene20_clone_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene20_clone_pose = scene20_clone_pose[scene20_clone_pose['cameraID']==1][['angle','zloc']]

        scene20_clone_bbox.reset_index(inplace=True)
        scene20_clone_pose.reset_index(inplace=True)
        scene20_clone_bbox.drop('index', axis=1, inplace=True)
        scene20_clone_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene20_clone_data = pd.DataFrame({'filename':filename})
        scene20_clone_data['class'] = label_list
        scene20_clone_data = pd.concat([scene20_clone_data, scene20_clone_bbox], axis=1)
        scene20_clone_data = pd.concat([scene20_clone_data, scene20_clone_pose], axis=1)
        scene20_clone_data['weather'] = 'clone'
        scene20_clone_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene20_clone[length*i:length*(i+1)]
        mask = [index for index in range(len(scene20_clone_data)) if scene20_clone_data['filename'].values[index] in set_filename]
        scene20_clone_data = scene20_clone_data.iloc[mask]

        # 데이터 최종 병합
        scene20_vkitti_data = pd.concat([scene20_vkitti_data, scene20_clone_data], axis=0)
        
    # 'fog'
    elif i == 1:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene20_fog_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene20_fog_bbox = scene20_fog_bbox[scene20_fog_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene20_fog[fr] for fr in scene20_fog_bbox['frame'].values]
        scene20_fog_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene20_fog_info[scene20_fog_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene20_fog_bbox['trackID'].values] 
        scene20_fog_bbox.drop('trackID', axis=1, inplace=True)

        scene20_fog_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene20_fog_pose = scene20_fog_pose[scene20_fog_pose['cameraID']==1][['angle','zloc']]

        scene20_fog_bbox.reset_index(inplace=True)
        scene20_fog_pose.reset_index(inplace=True)
        scene20_fog_bbox.drop('index', axis=1, inplace=True)
        scene20_fog_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene20_fog_data = pd.DataFrame({'filename':filename})
        scene20_fog_data['class'] = label_list
        scene20_fog_data = pd.concat([scene20_fog_data, scene20_fog_bbox], axis=1)
        scene20_fog_data = pd.concat([scene20_fog_data, scene20_fog_pose], axis=1)
        scene20_fog_data['weather'] = 'fog'
        scene20_fog_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene20_fog[length*i:length*(i+1)]
        mask = [index for index in range(len(scene20_fog_data)) if scene20_fog_data['filename'].values[index] in set_filename]
        scene20_fog_data = scene20_fog_data.iloc[mask]

        # 데이터 최종 병합
        scene20_vkitti_data = pd.concat([scene20_vkitti_data, scene20_fog_data], axis=0)
        
    # 'morning'
    elif i == 2:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene20_morning_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene20_morning_bbox = scene20_morning_bbox[scene20_morning_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene20_morning[fr] for fr in scene20_morning_bbox['frame'].values]
        scene20_morning_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene20_morning_info[scene20_morning_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene20_morning_bbox['trackID'].values] 
        scene20_morning_bbox.drop('trackID', axis=1, inplace=True)

        scene20_morning_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene20_morning_pose = scene20_morning_pose[scene20_morning_pose['cameraID']==1][['angle','zloc']]

        scene20_morning_bbox.reset_index(inplace=True)
        scene20_morning_pose.reset_index(inplace=True)
        scene20_morning_bbox.drop('index', axis=1, inplace=True)
        scene20_morning_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene20_morning_data = pd.DataFrame({'filename':filename})
        scene20_morning_data['class'] = label_list
        scene20_morning_data = pd.concat([scene20_morning_data, scene20_morning_bbox], axis=1)
        scene20_morning_data = pd.concat([scene20_morning_data, scene20_morning_pose], axis=1)
        scene20_morning_data['weather'] = 'morning'
        scene20_morning_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene20_morning[length*i:length*(i+1)]
        mask = [index for index in range(len(scene20_morning_data)) if scene20_morning_data['filename'].values[index] in set_filename]
        scene20_morning_data = scene20_morning_data.iloc[mask]

        # 데이터 최종 병합
        scene20_vkitti_data = pd.concat([scene20_vkitti_data, scene20_morning_data], axis=0)
        
    # 'overcast'
    elif i == 3:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene20_overcast_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene20_overcast_bbox = scene20_overcast_bbox[scene20_overcast_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene20_overcast[fr] for fr in scene20_overcast_bbox['frame'].values]
        scene20_overcast_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene20_overcast_info[scene20_overcast_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene20_overcast_bbox['trackID'].values] 
        scene20_overcast_bbox.drop('trackID', axis=1, inplace=True)

        scene20_overcast_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene20_overcast_pose = scene20_overcast_pose[scene20_overcast_pose['cameraID']==1][['angle','zloc']]

        scene20_overcast_bbox.reset_index(inplace=True)
        scene20_overcast_pose.reset_index(inplace=True)
        scene20_overcast_bbox.drop('index', axis=1, inplace=True)
        scene20_overcast_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene20_overcast_data = pd.DataFrame({'filename':filename})
        scene20_overcast_data['class'] = label_list
        scene20_overcast_data = pd.concat([scene20_overcast_data, scene20_overcast_bbox], axis=1)
        scene20_overcast_data = pd.concat([scene20_overcast_data, scene20_overcast_pose], axis=1)
        scene20_overcast_data['weather'] = 'overcast'
        scene20_overcast_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene20_overcast[length*i:length*(i+1)]
        mask = [index for index in range(len(scene20_overcast_data)) if scene20_overcast_data['filename'].values[index] in set_filename]
        scene20_overcast_data = scene20_overcast_data.iloc[mask]

        # 데이터 최종 병합
        scene20_vkitti_data = pd.concat([scene20_vkitti_data, scene20_overcast_data], axis=0)
        
    # 'rain'
    elif i == 4:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene20_rain_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene20_rain_bbox = scene20_rain_bbox[scene20_rain_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene20_rain[fr] for fr in scene20_rain_bbox['frame'].values]
        scene20_rain_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene20_rain_info[scene20_rain_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene20_rain_bbox['trackID'].values] 
        scene20_rain_bbox.drop('trackID', axis=1, inplace=True)

        scene20_rain_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene20_rain_pose = scene20_rain_pose[scene20_rain_pose['cameraID']==1][['angle','zloc']]

        scene20_rain_bbox.reset_index(inplace=True)
        scene20_rain_pose.reset_index(inplace=True)
        scene20_rain_bbox.drop('index', axis=1, inplace=True)
        scene20_rain_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene20_rain_data = pd.DataFrame({'filename':filename})
        scene20_rain_data['class'] = label_list
        scene20_rain_data = pd.concat([scene20_rain_data, scene20_rain_bbox], axis=1)
        scene20_rain_data = pd.concat([scene20_rain_data, scene20_rain_pose], axis=1)
        scene20_rain_data['weather'] = 'rain'
        scene20_rain_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene20_rain[length*i:length*(i+1)]
        mask = [index for index in range(len(scene20_rain_data)) if scene20_rain_data['filename'].values[index] in set_filename]
        scene20_rain_data = scene20_rain_data.iloc[mask]

        # 데이터 최종 병합
        scene20_vkitti_data = pd.concat([scene20_vkitti_data, scene20_rain_data], axis=0)
        
    # 'sunset'
    elif i == 5:
        # bbox 데이터에서 xmin, ymin, xmax, ymax value 뽑아내기
        # camera_ID == 1만 사용
        scene20_sunset_bbox.rename(columns={'left':'xmin', 'right':'xmax', 'top':'ymin', 'bottom':'ymax'}, inplace=True)
        scene20_sunset_bbox = scene20_sunset_bbox[scene20_sunset_bbox['cameraID']==1][['xmin','ymin','xmax','ymax','trackID','frame']]

        filename = [scene20_sunset[fr] for fr in scene20_sunset_bbox['frame'].values]
        scene20_sunset_bbox.drop('frame', axis=1, inplace=True)

        # trackID에 맞는 label을 대응
        label_list = [scene20_sunset_info[scene20_sunset_info['trackID']==trackID]['label'].values[0] \
                      for trackID in scene20_sunset_bbox['trackID'].values] 
        scene20_sunset_bbox.drop('trackID', axis=1, inplace=True)

        scene20_sunset_pose.rename(columns={'camera_space_Z':'zloc', 'alpha':'angle'}, inplace=True)
        scene20_sunset_pose = scene20_sunset_pose[scene20_sunset_pose['cameraID']==1][['angle','zloc']]

        scene20_sunset_bbox.reset_index(inplace=True)
        scene20_sunset_pose.reset_index(inplace=True)
        scene20_sunset_bbox.drop('index', axis=1, inplace=True)
        scene20_sunset_pose.drop('index', axis=1, inplace=True)

        # 데이터 병합하기
        scene20_sunset_data = pd.DataFrame({'filename':filename})
        scene20_sunset_data['class'] = label_list
        scene20_sunset_data = pd.concat([scene20_sunset_data, scene20_sunset_bbox], axis=1)
        scene20_sunset_data = pd.concat([scene20_sunset_data, scene20_sunset_pose], axis=1)
        scene20_sunset_data['weather'] = 'sunset'
        scene20_sunset_data['class'].replace({'Car':'car', 'Van':'car', 'Truck':'truck'}, inplace=True)

        set_filename = scene20_sunset[length*i:length*(i+1)]
        mask = [index for index in range(len(scene20_sunset_data)) if scene20_sunset_data['filename'].values[index] in set_filename]
        scene20_sunset_data = scene20_sunset_data.iloc[mask]

        # 데이터 최종 병합
        scene20_vkitti_data = pd.concat([scene20_vkitti_data, scene20_sunset_data], axis=0)
        
scene20_vkitti_data['filename'] = ['scene20_'+name for name in scene20_vkitti_data['filename']]

scene20_vkitti_data.isnull().sum(axis=0)
#scene20_vkitti_data.to_csv('./scene20_vkitti_data.csv', mode='a', index=False)

print('scene20 end')

print('Finish')
end = time.time() # 시간 측정 끝
print(f"{end - start:.5f} sec") #29.84161 sec


# 데이터 최종 병합
vkitti_annotations = pd.DataFrame()
for data in [scene1_vkitti_data, scene2_vkitti_data, scene6_vkitti_data, scene18_vkitti_data, scene20_vkitti_data]:
    vkitti_annotations = pd.concat([vkitti_annotations, data], axis=0)
vkitti_annotations.to_csv('../vkitti_annotations.csv', mode='a', index=False)