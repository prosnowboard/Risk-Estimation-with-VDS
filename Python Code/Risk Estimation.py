# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:57:06 2018

@author: Hyunjin
"""
# 2018.08.03 Refactoring 중 입니다.

#%%
# 0. 사용할 패키지 불러오기
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dask.dataframe as dd
import glob
import time
#from keras import backend as K

#%% Input Data settting
# att 파일
# INPUT_FILENAME = att_FileList[i]
# i = 0 + 1

def InputDataSetting(INPUT_FILENAME):
    
    ddf = dd.read_csv(INPUT_FILENAME, skiprows = 16, delimiter = ';')
    
    # ddf.head()
    
    input_dataset1 = ddf.fillna(0)
    
    input_dataset2 = input_dataset1.compute()
    
    # simulation filtering
    
    SimRun = input_dataset2['$DATACOLLECTIONMEASUREMENTEVALUATION:SIMRUN'].iloc[-1].astype(int)
    
    input_dataset2 = input_dataset2[input_dataset2['$DATACOLLECTIONMEASUREMENTEVALUATION:SIMRUN'] == SimRun]
    
    input_dataset2['TIME_E'] = input_dataset2['TIMEINT'].str.split('-', 1).str[1].astype(int)
    
    input_dataset2 = input_dataset2.pivot_table(index='TIME_E', columns = 'DATACOLLECTIONMEASUREMENT', values = {'VEHS(ALL)', 'SPEED(ALL)'})
    
    DIFF_NAME = ['1-4', '2-5', '3-6', '4-7', '5-8', '6-9', '7-10',
                 '8-11','9-12','10-13','11-14','12-15','13-16',
                 '14-17','15-18','16-19','17-20','18-21']
    
    
    # DIFF_NAME 개수가 18
    for j in range(18):
        
        input_dataset2[('SPEED(ALL)', DIFF_NAME[j])] = input_dataset2[('SPEED(ALL)', j + 1)] - input_dataset2[('SPEED(ALL)', j + 4)]
        input_dataset2[('VEHS(ALL)', DIFF_NAME[j])] = input_dataset2[('VEHS(ALL)', j + 1)] - input_dataset2[('VEHS(ALL)', j + 4)]

    return input_dataset2

#%% Output Data Setting
# fzp 파일
# OUTPUT_FILENAME = fzp_FileList[i]
    
def OutputDataSetting(OUTPUT_FILENAME):
        
    ddf = dd.read_csv(OUTPUT_FILENAME, skiprows = 16, delimiter = ';')
    
    # ddf.head()
    
    # Follow Dist - Safe Dist : 상충 판단
    ddf = ddf.assign(SDV = ddf.FOLLOWDIST - ddf.SAFEDIST)
    
    # 링크 2 Filter
    ddf = ddf[(ddf.LANE == '2-1') | (ddf.LANE == '2-2') | (ddf.LANE == '2-3')]
    
    # Safe Index 1(safe) or -1(s)
    ddf['SI'] = ddf.SDV.where(ddf.SDV > 0, -1) 
    ddf['SI'] = ddf.SI.where(ddf.SI < 0, 1)
    ddf['SI2'] = ddf['SI']

    
    '''
    The where method is an application of the if-then idiom. For each element in the calling DataFrame, if cond is True the element is used; otherwise the corresponding element from the DataFrame other is used.
    
    The signature for DataFrame.where() differs from numpy.where(). Roughly df1.where(m, df2) is equivalent to np.where(m, df1, df2).
    
    '''
        
    output_dataset = ddf.compute() 
    
    SIM_TIME = output_dataset['$VEHICLE:SIMSEC'].iloc[-1].astype(int)
    
    # index 구간 나누기
    index_section = pd.cut(output_dataset['$VEHICLE:SIMSEC'],  list(range(0, SIM_TIME + 1, 30)))
    
    output_dataset3 = output_dataset.pivot_table(index = index_section, columns = 'SI', values = 'SI2', aggfunc = {'count'})
    
    output_dataset3 = output_dataset3.fillna(0)

    return output_dataset3


#%% 데이터 크기 통일하기

#INPUT_DATASET = ResamplingInputDataSettting(INPUT_DATASET, OUTPUT_DATASET)

def ResamplingInputDataSettting(INPUT_DATASET, OUTPUT_DATASET):
        
    input_len = len(INPUT_DATASET)
    
    output_len = len(OUTPUT_DATASET)
        
    diff_len = input_len - output_len
    
    input_dataset3 = INPUT_DATASET.iloc[diff_len:,:]

    return input_dataset3


#%% 모델 구성하기
    
def SetModel(inputLayer, hiddenLayer, numHiddenLayer, outputLayer, optimizer, loss):
    
    # 모델 구성하기
    
    model = Sequential()
    
    model.add(inputLayer)
    
    for j in range(numHiddenLayer):
        
        model.add(hiddenLayer)
        
    model.add(outputLayer)
    
    # 모델 학습과정 설정하기
    
    model.compile(optimizer = optimizer, loss = loss)
    
    return model

#%% 모델 학습하기
    
def SetModelLearning(INPUT_DATASET, OUTPUT_DATASET, epochs, batch_size):
        
    # 데이터셋 생성하기
    # input dim 42로!!
    
    x_train, x_test, y_train, y_test = train_test_split(INPUT_DATASET,
                                                        OUTPUT_DATASET[('count', -1.0)],
                                                        test_size = 0.2,
                                                        random_state = 42)
    
    # 모델 학습시키기
    hist = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

    return x_train, x_test, y_train, y_test, hist

#%% 학습과정 살펴보기
def SetDrawPlot(hist):
    
    plt.plot(hist.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

#%% 모델 평가하기

def SetModelEvaluate(x_test, y_test):
    loss = model.evaluate(x_test, y_test, batch_size = 5)
    print('loss : ' + str(loss))
    return loss


#%% 파일 불러오기
def LoadFileList(folderName):
    
    # INPUT
    att_FileList = glob.glob(folderName + '/*.att') #Get folder path containing text files

    # OUTPUT
    fzp_FileList = glob.glob(folderName + '/*.fzp') #Get folder path containing text files
    
    return att_FileList, fzp_FileList
    
#%% MAPE 계산
    
def Cal_MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# %%===========================================================================
# Main Code
# =============================================================================
                           
att_FileList, fzp_FileList = LoadFileList(folderName = 'Dataset Sample')

#SIM_TIME = 14400

loss_list = []
DATASET_list = []

# 모델 구성(초기화)
model = SetModel(inputLayer = Dense(256, input_dim = 78, activation='relu')
                 , hiddenLayer = Dense(256, activation='relu')
                 , numHiddenLayer = 5
                 , outputLayer = Dense(1)
                 , optimizer='sgd'
                 , loss='mse')


# %%===========================================================================
# Model Learning !!
# =============================================================================

start_time = time.time()
#for i in range(len(FZP_FILE_LIST)):

numFiles = len(fzp_FileList)

for i in range(numFiles):
    
    print("Model Learning :")
    print(att_FileList[i])
    print(fzp_FileList[i])
    print(" i : ", i)
    
    INPUT_DATASET = InputDataSetting(att_FileList[i])
    
    OUTPUT_DATASET = OutputDataSetting(fzp_FileList[i])
    
    INPUT_DATASET = ResamplingInputDataSettting(INPUT_DATASET, OUTPUT_DATASET)
    
    x_train, x_test, y_train, y_test, hist = SetModelLearning(INPUT_DATASET, OUTPUT_DATASET
                                                              , epochs = 3000
                                                              , batch_size = 50)
    
    DATASET_list.append([x_train, x_test, y_train, y_test]) 
    
    SetDrawPlot(hist)
    
    loss_list.append(SetModelEvaluate(x_test, y_test))

print("--- %s seconds ---" % (time.time() - start_time))

#%% Loss Figure

plt.figure()    
plt.plot(loss_list)

#%% MAPE 계산

MAPE_list = []

for i in range(numFiles):
    
    x_test = DATASET_list[i][1]
    y_predict = model.predict(x_test)
    y_true = y_test[y_test>0]
    y_predict = y_predict.flatten()
    y_pred = y_predict[y_test>0]
    MAPE_list.append(Cal_MAPE(y_true, y_pred))

np.mean(MAPE_list)

#%% 예측 정확도 그래프 

# i : 파
i = 0

# x_train = DATASET_list[i][0]
# x_test = DATASET_list[i][1]
# y_train = DATASET_list[i][2]
# y_test = DATASET_list[i][3]

x_test = DATASET_list[i][1]

y_predict = model.predict(x_test)

plt.plot(y_test.values, label = 'Observed')
plt.plot(y_predict, label = 'Predicted')

plt.xlabel('Case', fontsize = 15)
plt.ylabel('Conflict', fontsize = 15)

plt.legend()

#%% 학습모델 저장 및 불러오기

# 모델 저장
# model.save('20180731_4A.h5')

# 모델 불러오기
# from keras.models import load_model
# model = load_model('20180731_isthattrue_model.h5')


#%% Observed 오름차순 후 Bar 그래프 비교

dataset = pd.DataFrame({'Observed' : y_test.values, 'Predicted':y_predict.flatten()})
dataset = dataset.sort_values(by='Observed')
ax = dataset.plot.bar(rot=0)
ax.set_xlabel("Data Case")
ax.set_ylabel("Conflict")
plt.show()

#%%
import dill                            #pip install dill --user
filename = 'isthattrue.pkl'
dill.dump_session(filename)

# and to load the session again:
dill.load_session(filename)

#%% 참조
'''

*** 2017.07.31 ***

2A : 약 30분
교통량 증가 A->D (3600초마다)(총 14,400초 * 25개 파일)
epochs = 2000, batch_size = 20
model.add(Dense(10, input_dim = 78, activation='relu'))
for j in range(2):
    model.add(Dense(10, activation='relu'))
MAPE : 464.27418376883827


3A : 약 1시간
교통량 증가 A->D (3600초마다)(총 14,400초 * 25개 파일)
epochs = 2000, batch_size = 20
model.add(Dense(128, input_dim = 78, activation='relu'))
for j in range(7):
    model.add(Dense(256, activation='relu'))
MAPE : 70.868595821660179    
    
4A : 약 4시
교통량 증가 A->D (3600초마다)(총 14,400초 * 25개 파일)
epochs = 2000, batch_size = 20
model.add(Dense(256, input_dim = 78, activation='relu'))
for j in range(19):
    model.add(Dense(256, activation='relu'))
MAPE : 40.237679268897814    
    
5A : 약 10시간
#1 LOSS nan 오류
교통량 증가 A->D (3600초마다)(총 14,400초 * 25개 파일)
epochs = 2000, batch_size = 20
model.add(Dense(256, input_dim = 78, activation='relu'))
for j in range(49):
    model.add(Dense(256, activation='relu')
MAPE :     
    


8A : 약 1시간 반     
교통량 증가 A->D (3600초마다)(총 14,400초 * 125개 파일)
epochs = 2000, batch_size = 100
model.add(Dense(20, input_dim = 78, activation='relu'))
for j in range(9):
    model.add(Dense(20, activation='relu'))
MAPE : 34.75421719641767        
   
9A : 약 4시간    
교통량 증가 A->D (3600초마다)(총 14,400초 * 125개 파일)
epochs = 3000, batch_size = 50
model.add(Dense(128, input_dim = 78, activation='relu'))
for j in range(40):
    model.add(Dense(128, activation='relu'))    
'''



