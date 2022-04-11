# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:02:14 2022

@author: ODD
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets.custom_datasets import CustomDataset


df = pd.read_csv('./datasets/glp_kitti_data.csv')

variable = ['xmin','ymin','xmax','ymax','depth_min']
dataset = CustomDataset(df, variable)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

for i, (data, labels) in enumerate(dataloader):
    print(data)
    print('data shape:', data.size())
    print(labels)
    print('label shape:',labels.size())
    
    if i == 3:
        break