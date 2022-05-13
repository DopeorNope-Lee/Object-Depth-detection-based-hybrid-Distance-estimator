from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from model.detr import DETR
from model.glpdepth import GLP
#from odd.gtts_mp3 import Warninggtts
import xgboost as xgb
import warnings
from PIL import Image
from scipy import stats
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################# Start ###########################
#warnings.filterwarnings(action='ignore')

"""
xgb_file="./odd/weights/lastxgb"
detr_model=DETR()
glp_model=GLP()
xgb_model=xgb.XGBRegressor()
xgb_model.load_model(xgb_file)
"""

'''
function define
'''
#warn=Warninggtts("warningmessage")
#warn.saving_speaking("물체가 근접합니다")
#warn1=Warninggtts("errorgmessage")
#warn1.saving_speaking("시스템이 정상 작동 하지 않습니다")
count = 0
# 속력 측정
def speed_estimate(prev,current_v):
    diff=(prev-current_v)/1000
    velocity= diff*3600
    return velocity

def odd_process(zloc,speed, count):
    if count == 1: 
        if speed>=80: #여기서 스피드는 속력이 아니라 속도(상대적 속도임)
            if zloc<50:
                warn.speak()
        elif speed>=40:
            if zloc<30:
                warn.speak()
        elif speed>=10:
            if zloc<10:
                warn.speak()
                
                
'''
Model 및 카메라 정의
'''
# 모델 정의
# DETR 불러오기
model_path = 'facebookresearch/detr:main'
model_backbone = 'detr_resnet101'
#sys.modules.pop('models') # ModuleNotFoundError: No module named 'models.backbone' 이 에러 발생시 수행
DETR = DETR(model_path, model_backbone)
DETR.model.eval()
DETR.model.to(device)

# GLPdepth 불러오기
glp_pretrained = 'vinvino02/glpn-kitti'
GLPdepth = GLP(glp_pretrained)
GLPdepth.model.eval()
GLPdepth.model.to(device)

# 카메라 정의
cap = cv2.VideoCapture('./test_video/object_video1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
os.makedirs('./test_video/output', exist_ok=True)
os.makedirs('./test_video/frame', exist_ok=True)
out = cv2.VideoWriter('./test_video/output/ODD_test.mp4', fourcc, 30.0, (1242,374))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1242) # 가로
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 374) # 세로


'''
# 비디오 작동하기
'''
currentframe = 1
if cap.isOpened():
    while(True):
        start = time.time() # 시간 측정 시작
        ret, frame= cap.read()
        if ret:
            #cv2.imshow("webcam",frame)
            
            # 테스트를 위해 임시로 넣음.
            name = './test_video/frame/object_video1_'+str(currentframe)+'.jpg'
            
            if cv2.waitKey(1) != -1:
                #cv2.imwrite('webcam_snap.jpg',frame)
                break
            #정상적인 케이스임
            #first_step = detr_model(frame)
            #second_step =GLPdepth(frame,first_step)
            #speed="계산 방법"
            #zloc= xgb_model.predict("여기서는 들어가는 최종 텐서를 맞추어서 넣어주면됨.")
            #odd_process(zloc,speed)
            
            cv2.imwrite(name, frame)
            currentframe += 1
            
            '''
            Step1) Image DETR 적용
            '''
            frame = cv2.resize(frame, (1242, 374))
            color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_coverted)
            img_shape = color_coverted.shape[0:2]
            
            # Predicted
            scores, boxes = DETR.detect(pil_image) # Detection
            
            
            '''
            Step2) GLP_Depth 적용
            '''
            # Make depth map
            prediction = GLPdepth.predict(pil_image, img_shape)
            
            
            '''
            Step3) 입력
            '''
            # BBOX input
            k = 1
            xmin_list = [] ; ymin_list = [] ; xmax_list = [] ; ymax_list = []
            for p, (xmin, ymin, xmax, ymax) in zip(scores, boxes.tolist()):
                prt = True
                
                #print(xmin, ymin, xmax, ymax)
                cl = p.argmax()
                
                # Variable 'p' setting => no gradient
                classes = DETR.CLASSES[cl]
                if classes not in ['person', 'truck', 'car', 'bicycle', 'train']:
                    continue
                else:
                    cl = ['person','truck','car','bicycle','train'].index(classes)
                    
                # rgb
                r,g,b = DETR.COLORS[cl][0] * 255, DETR.COLORS[cl][1] * 255, DETR.COLORS[cl][2] * 255
                rgb = (r,g,b)
                
                # depth map의 index는 최소한 0
                if int(xmin) < 0:
                    xmin = 0
                if int(ymin) < 0:
                    ymin = 0
                    
                depth_mean = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].mean()
                depth_median = np.median(prediction[int(ymin):int(ymax),int(xmin):int(xmax)])
                depth_mean_trim = stats.trim_mean(prediction[int(ymin):int(ymax), int(xmin):int(xmax)].flatten(), 0.2)
                depth_min = prediction[int(ymin):int(ymax),int(xmin):int(xmax)].min()
                xy = np.where(prediction==depth_min)
                
                '''
                전처리
                bbox 비교해서 70% 이상 겹친다면 그 뒤에 있는 영역을 지우고,
                만약 아니라면, 겹친 부분을 제외한 후, 다시 depth를 계산해서 값 출력
                '''
                xmin_list.insert(0,xmin) ; ymin_list.insert(0,ymin) ; 
                xmax_list.insert(0,xmax) ; ymax_list.insert(0,ymax) ;
                #print(ymin_list)
                
                if k == 1:
                    k += 1
                    continue
                    
                elif k >= 2: 
                    for i in range(len(xmin_list)-1):
                        y_range1 = np.arange(int(ymin_list[0]), int(ymax_list[0]+1))
                        y_range2 = np.arange(int(ymin_list[i+1]), int(ymax_list[i+1]+1))
                        y_intersect = np.intersect1d(y_range1, y_range2)
                        
                        #print(y_intersect)
                        
                        if len(y_intersect) >= 1: 
                            x_range1 = np.arange(int(xmin_list[0]), int(xmax_list[0])+1)
                            x_range2 = np.arange(int(xmin_list[i+1]), int(xmax_list[i+1]+1))
                            x_intersect = np.intersect1d(x_range1, x_range2)
                            
                            #print(x_intersect)
                            
                            if len(x_intersect) >= 1: # BBOX가 겹친다면 밑에 구문 실행
                                area1 = (y_range1.max() - y_range1.min())*(x_range1.max() - x_range1.min())
                                area2 = (y_range2.max() - y_range2.min())*(x_range2.max() - x_range2.min())
                                area_intersect = (y_intersect.max() - y_intersect.min())*(x_intersect.max() - x_intersect.min())
                                
                                if area_intersect/area1 >= 0.70 or area_intersect/area2 >= 0.70: # 70% 이상 면적을 공유한다면
                                    prt = False # 출력 안함
                                    continue
                                    
                                # 조금 겹친다면 depth_min and depth_mean 값 수정
                                elif  area_intersect/area1 > 0 or area_intersect/area2 > 0:
                                    if area1 < area2:
                                        prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                        bbox = prediction[int(ymin_list[0]):int(ymax_list[0]), int(xmin_list[0]):int(xmax_list[0])]
                                        depth_min  = np.nanmin(bbox)
                                        depth_mean = np.nanmean(bbox)
                                        
                                    else:
                                        prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                        bbox = prediction[int(ymin_list[i+1]):int(ymax_list[i+1]), int(xmin_list[i+1]):int(xmax_list[i+1])]
                                        depth_min  = np.nanmin(bbox)
                                        depth_mean = np.nanmean(bbox)
                                        
                                    
                    # input text & draw bbox
                    if prt == True: 
                        # error1: 좌표는 int형.
                        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), rgb, 3)
                        
                        cv2.putText(frame, classes+' '+str(round(depth_mean,1)), (int(xmin)-5, int(ymin)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 1,
                                    lineType=cv2.LINE_AA)
                                    

            cv2.imshow('video1', frame)
            # Save Video (depth image)
            out.write(frame)

        else:
            print("프레임을 받을 수 없습니다.")
            #warn1.speak()
            break
        
        count = 1
        end = time.time() # 시간 측정 끝
        print(f"{end - start:.5f} sec") # each frame: 
        
else:
    print('파일을 열 수 없습니다')
    #warn1.speak()
    
# OpenCV 중지
cap.release()
out.release()
cv2.destroyAllWindows()   

