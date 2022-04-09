# Object-Depth-detection-based-hybrid-Distance-estimator
비디오를 활용한 트랜스포머 Depth 측정모델을 융합한 거리측정 하이브리드 모델



# 1. 데이터셋 다운 받기
구글드라이브에 접속 후, data.egg 파일을 프로젝트 폴더안의'./datasets/data/' 경로에 다운  
폴더의 'image', 'VKITTI', 'VKITTI_txt' 이렇게 3개가 나오도록 설정.

# 2. 데이터셋 Preprocessing
i) KITTI
1. './kitti_detr_dataset.py'  -- kitti dataset이 나타내는 bbox와 DETR로 kitti image를 예측해서 나온 bbox가 무엇인지 비교하교, 실제 데이터의 bbox가 가르킨 객체를 DETR이 어떤 bbox로 표현해서 예측했는지 비교해서 DETR의 bbox만 저장하기.  
2. './kitti_glpdepth_dataset.py'  -- glpdepth로 kitti image를 Depth map으로 표현 후, 위에서 나온 bbox로 Depth map을 잘라서(bbox 모양의 Depth map을 추출) 그 Depth map의 value(method: min, mean, ...)를 얻는다.  

ii) VKITTI2  
1. './datasets/data/make_vkitti_dataset.py'  -- VKITTI 데이터 안에 있는 모든 Scene과 각각의 weather 정보에 대해서 데이터 정리  
