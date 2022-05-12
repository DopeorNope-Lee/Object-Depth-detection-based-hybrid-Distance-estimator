from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from model.detr import*
from model.glpdepth import*
from odd.gtts_mp3 import Warninggtts
import xgboost as xgb
import warnings



############################# Start ###########################
#warnings.filterwarnings(action='ignore')

"""
xgb_file="./odd/weights/lastxgb"
detr_model=DETR()
glp_model=GLP()
xgb_model=xgb.XGBRegressor()
xgb_model.load_model(xgb_file)
"""
warn=Warninggtts("warningmessage")
warn.saving_speaking("물체가 근접합니다")
warn1=Warninggtts("errorgmessage")
warn1.saving_speaking("시스템이 정상 작동 하지 않습니다")


def odd_process(zloc,speed):
    if speed>=80: #여기서 스피드는 속력이 아니라 속도(상대적 속도임)
        if zloc<50:
            warn.speak()
    elif speed>=40:
        if zloc<30:
            warn.speak()
    elif speed>=10:
        if zloc<10:
            warn.speak()




cap=cv2.VideoCapture(0)
if cap.isOpened():
    while(True):
        ret, frame= cap.read()
        if ret:
            cv2.imshow("webcam",frame)
            if cv2.waitKey(1) != -1:
                cv2.imwrite('webcam_snap.jpg',frame)
                break
            #정상적인 케이스임
            #first_step = detr_model(frame)
            #second_step =GLPdepth(frame,first_step)
            #speed="계산 방법"
            #zloc= xgb_model.predict("여기서는 들어가는 최종 텐서를 맞추어서 넣어주면됨.")
            #odd_process(zloc,speed)
            
        else:
            print("프레임을 받을 수 없습니다.")
            warn1.speak()
            break
else:
    print('파일을 열 수 없습니다')
    warn1.speak()
    

cap.release()
cv2.destroyAllWindows()
