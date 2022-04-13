# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:08:07 2022

@author: ODD
"""

import pandas as pd

glp_kitti_data = pd.read_csv('./glp_kitti_data.csv')
glp_kitti_data.info()

# train: 70%, valid: 15%, test: 15%
kitti_length = len(glp_kitti_data)
train_len = int(kitti_length*0.7)
valid_len = int(kitti_length*0.15)
test_len = int(kitti_length*0.15)

kitti_train = glp_kitti_data.iloc[:train_len,:]
kitti_valid = glp_kitti_data.iloc[train_len:(train_len+valid_len),:]
kitti_test = glp_kitti_data.iloc[(train_len+valid_len):,:]

# width, height 열 추가
kitti_train['width'] = kitti_train['xmax'] - kitti_train['xmin']
kitti_train['height'] = kitti_train['ymax'] - kitti_train['ymin']
kitti_valid['width'] = kitti_valid['xmax'] - kitti_valid['xmin']
kitti_valid['height'] = kitti_valid['ymax'] - kitti_valid['ymin']
kitti_test['width'] = kitti_test['xmax'] - kitti_test['xmin']
kitti_test['height'] = kitti_test['ymax'] - kitti_test['ymin']

# 저장
#kitti_train.to_csv('./kitti_train.csv', mode='a', index=False)
#kitti_valid.to_csv('./kitti_valid.csv', mode='a', index=False)
#kitti_test.to_csv('./kitti_test.csv', mode='a', index=False)

# KITTI_VKITTI (7:1.5:1.5)
glp_vkitti_data = pd.read_csv('./glp_vkitti_data.csv')
vkitti_length = len(glp_vkitti_data)
train_len = int(vkitti_length*0.7)
valid_len = int(vkitti_length*0.15)
test_len = int(vkitti_length*0.15)

vkitti_train = glp_vkitti_data.iloc[:train_len,:]
vkitti_valid = glp_vkitti_data.iloc[train_len:(train_len+valid_len),:]
vkitti_test = glp_vkitti_data.iloc[(train_len+valid_len):,:]

# width, height 열 추가
vkitti_train['width'] = vkitti_train['xmax'] - vkitti_train['xmin']
vkitti_train['height'] = vkitti_train['ymax'] - vkitti_train['ymin']
vkitti_valid['width'] = vkitti_valid['xmax'] - vkitti_valid['xmin']
vkitti_valid['height'] = vkitti_valid['ymax'] - vkitti_valid['ymin']
vkitti_test['width'] = vkitti_test['xmax'] - vkitti_test['xmin']
vkitti_test['height'] = vkitti_test['ymax'] - vkitti_test['ymin']

# VKITTI와 KITTI 
vkitti_train = pd.concat([vkitti_train, kitti_train])
vkitti_valid = pd.concat([vkitti_valid, kitti_valid])
vkitti_test = pd.concat([vkitti_valid, kitti_test])

# 저장
vkitti_train.to_csv('./vkitti_kitti_train.csv', mode='a', index=False)
vkitti_valid.to_csv('./vkitti_kitti_valid.csv', mode='a', index=False)
vkitti_test.to_csv('./vkitti_kitti_test.csv', mode='a', index=False)


