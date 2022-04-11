# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:08:07 2022

@author: ODD
"""

import pandas as pd
import shutil # image copy
import numpy as np
from sklearn.model_selection import train_test_split

glp_kitti_data = pd.read_csv('./glp_kitti_data.csv')
glp_kitti_data.info()

# train: 70%, valid: 70%, test: 10%
length = len(glp_kitti_data)
train_len = length*0.7
valid_len = length*0.2
test_len = length*0.1

kitti_train = glp_kitti_data.iloc[:22022,:]
kitti_valid = glp_kitti_data.iloc[22022:28314,:]
kitti_test = glp_kitti_data.iloc[28314:,:]

# 저장
kitti_train.to_csv('./kitti_train.csv', mode='a', index=False)
kitti_valid.to_csv('./kitti_valid.csv', mode='a', index=False)
kitti_test.to_csv('./kitti_test.csv', mode='a', index=False)

