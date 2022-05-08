from warnings import WarningMessage
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
#from custom_datasets import CustomDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from gtts import gTTS
import playsound
import cv2
from model.detr import*
from model.glpdepth import*
from odd.gtts_mp3 import Warninggtts


############################# Start ###########################

cap=cv2.VideoCapture(0)
warn=Warninggtts("warningmessage.mp3")

warn.speak()

if cap.isOpened():
    while(True):
        ret, frame= cap.read()
        if ret:
            cv2.imshow('webcam',frame)
            if cv2.waitKey(1) != -1:
                cv2.imwrite('webcam_snap.jpg',frame)
                break
            
        else:
            print("Can't reaceive the frame...")
            break
else:
    print('파일을 열 수 없습니다')
    

cap.release()
cv2.destroyAllWindows()
