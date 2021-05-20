#--------------------------------------------------------------------#
# 순천향대학교
# 빅데이터공학과
# 20171483 Han Tae Gyu
# 
# 머신러닝 HW_8
#--------------------------------------------------------------------#


#--------------------------------------------------------------------#
# use modules
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
#--------------------------------------------------------------------#

# (0) Problem (1 points) : 3~4 lines
# 	- Definition of Input, Output
#    

# 	- Is it a classificaiton? regression?
#      regression 문제입니다.
#      Victims를 분류하는 것이 아닌 Victims의
#      인원수를 예측해야 하는 문제 이기 때문 입니다.

# 	- How many number of labels?
#     Data shape : (10339, 15)
#     데이터의 label의 개수는 15개 입니다.

# 	- How the data looks like?
#       - Id ( 문자열 ) : 고유 ID
#       - District name ( 문자열 ) : 지역 이름
#       - Neighborhood Name ( 문자열 ) : 이웃 이름
#       - Street ( 문자열 ) : 거리이름
#       - Weekday ( 문자열 ) : 요일
#       - Month ( 문자열 ) : 월
#       - Day ( 숫자 ) : 날
#       - Hour ( 숫자 ) : 시각
#       - Part of the day ( 문자열 ) : 하루 중 언제 ( 오전, 오후, 밤 )
#       - Mild injuries  ( 숫자 ) : 가벼운 부상
#       - Serious injuries ( 숫자 ) : 심각한 부상
#       - Victims ( 숫자 ) : 희생자
#       - Vehicles involved ( 숫자 ) : 관련 차량
#       - Longitude ( 실수 ) : 경도
#       - Latitude  ( 실수 ) : 위도


# (1) Feature (1 points) : 3~4 lines
# 	- You can choose or define features by yourself
# 	- How did you preprocess the data?
#       희생자의 숫자를 예측하기 위해
#       Serious injuries, Mild injuries 
#       두개의 feature를  합쳐 새로운 
#       feature를 만들었습니다.

original_data = pd.read_csv('./data/accidents_2017.csv')
feature = pd.DataFrame({ "injuries_sum" : original_data[["Serious injuries", "Mild injuries"]].sum(axis=1)})
label = pd.DataFrame({ "Victims" : original_data["Victims"]})

# (2) Model (2 points) : 3~4 lines
# 	- You have to use one of the following list
# 	- Why did you choose your models?
#      choose models > Linear regression
#      이유 : 1개의 feature와 1개의 label을 가진 모델을
#      제작 하기 때문에 단순한 선형모델이 필요


# (3) Measure (2 points) : 3~4 lines
# 	- Explain the steps of measurement
# 	- 10-fold cross validation 
# 	- Classification: "weighted F1 score"
# 	- Regression: "MAE" (mean absolute error) : 선택


# 10-fold cross validation 객체 생성
kf10 = KFold(n_splits=10, shuffle=True) 

# 선형 회귀 모델 객체 생성             모델 파라미터
regr = linear_model.LinearRegression(fit_intercept=False,
                                     normalize=False,
                                     copy_X=True,
                                     n_jobs=None)

# MAE의 값들을 넣을 list 생성
kf10_MAE_list = []

i = 0

# 10-fold cross validation 동작 for문
for train_index, test_index in kf10.split(feature):
    i += 1
    X_train, X_test = feature.iloc[train_index,:], feature.iloc[test_index,:]
    Y_train, Y_test = label.iloc[train_index], label.iloc[test_index]
    
    regr.fit(X_train, Y_train) # 모델 훈련
    
    Y_pred = regr.predict(X_test) # 모델 예측
    MAE = mean_absolute_error(Y_test, Y_pred) # MAE 값 추출
    kf10_MAE_list.append(MAE) # MAE 값 list에 넣기
    print("fold {}: MAE = {}".format(i, MAE)) # MAE 값 출력
    
# MAE 값들 평균으로 출력 
MAE_avg = sum(kf10_MAE_list) / len(kf10_MAE_list)
print("Total (Average) MAE = {}".format(MAE_avg))


# (4) Model parameter engineering (4 points)
# 			- How did you change parameters to improve performance?
# 				- Explain the reason
# 				- (ex) I changed 'C' value to 100 because ....

# regr = linear_model.LinearRegression( fit_intercept=False,  > 절편의 계산여부 결정 모델의 feature와 label을 고려했을 때
#                                                               x,y축 (0,0)을 지나는 모델을 요구 따라서 > False

#                                       normalize=False,  > fit_intercept가 Flase이면 자동으로 False 설정

#                                       copy_X=True, #  > True로 설정하면 입력한 X가 함수 내에서 사용 할 수 있도록 복사 가능

#                                       n_jobs=None) # >  동시 프로세스 또는 동시 프로세스 수를 지정하는 데 사용
#                                                        학교 컴퓨터의 사양과 작동중인 프로그램을 알지 못함으로
#                                                        None으로 지정


# 			- Check the data imbalance, data sparsity, ...
# 				- Explain how you handel this
# 				- Note that high performance is not always good; for example, you may get 90~95% because of severe data imbalance. How can we handle this problem?

#       > 데이터 불균형
#          accidents_2017.csv 데이터를 가지고 예를 들면 
#          Victims의 데이터의 분포가 아래와 같습니다.

#          Victims    count
#          0            902
#          1           7385
#          2           1611
#          3            284
#          4            102
#          5             32
#          6             11
#          7              8
#          8              1
#          9              1
#          10             2  

#          데이터는 0 ~ 4 사이에 몰려있고 5 ~ 10에는
#          적게 존재합니다. 비율로만 봐도 70%가 넘는 엄청난
#          불균형 입니다. 이렇게 되면 과적합이 일어날 수 있습니다.
#          
#          과적합이 일어나는 이유는 데이터의 분포도가 높은 쪽에 
#          가중치를 적은 쪽보다 많이 두기 때문에 새로운 데이터가
#          들어오면 가중치가 높았던 쪽으로만 예측하기 때문에 예측을
#          0 ~ 4 사이로만 가능성이 높은 모델이 나오는 것입니다.
#          따라서 불균형성을 해결해야 합니다.

#          Under Sampling, Over Sampling 크게 두가지로  나누는데

#          Under Sampling은 많은 데이터를 가지고 있는 0 ~ 4의 데이터들을
#          5 ~ 10으로 맞추는 것을 말합니다. 
#          장점은 기존의 데이터를 깨끗한  상태로 놔둘 수 있습니다. 
#          단점은 데이터가 유실 됩니다.

#          Over Sampling은 적은 데이터를 가지고 있는 5 ~ 10의 데이터를
#          0 ~ 4로 맞추는 것을 말합니다. 
#          장점은 기존 데이터를 삭제하지 않습니다.
#          단점은 직접 얻은 데이터가 아니라 제작한
#          데이터이기 때문에 모델이 완벽하게 깨끗한 데이터는
#          아닙니다. 

#          이번 사례에서는 Under Sampling은 많은 데이터가 유실되고
#          모델을 훈련시키지 못하기 때문에 데이터가 적은 5 ~ 10 쪽을
#          Over Sampling을 해야합니다.