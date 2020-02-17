# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:33:08 2019

@author: Tazrin
"""

import pandas as pd
import numpy as np
import sklearn.metrics
from statistics import mean

data = pd.read_csv("TrainData.csv")

posCount = data[data.StudentPerformance == 'Positive'].count()["gender"]
negCount = data[data.StudentPerformance == 'Negative'].count()["gender"]
total = data.count()["gender"]
#prior calculation
Ppos = posCount/total
Pneg = negCount/total

gender = data.gender.unique()
stage = data.StageID.unique()
semester = data.Semester.unique()

for i in data.columns:
    data.columns.unique()
m = len(gender)
p = 1/m

#likelihood calculation of each feature
pgender = []
for i in gender:
    pos = ((data[(data.gender == i) & (data.StudentPerformance == 'Positive')].count()["gender"])+ m*p)/((posCount)+m)
    neg = ((data[(data.gender == i) & (data.StudentPerformance == 'Negative')].count()["gender"])+ m*p)/((negCount)+m)
    pgender.append([i,[pos,neg]])
    #pgender.append([i, pos])
    #pgender.append([i, neg])
pstage = []
m = len(stage)
p = 1/m
for i in stage:
    pos = ((data[(data.StageID == i) & (data.StudentPerformance == 'Positive')].count()["StageID"])+m*p)/((posCount)+m)
    neg = ((data[(data.StageID == i) & (data.StudentPerformance == 'Negative')].count()["StageID"])+m*p)/((negCount)+m)
    pstage.append([i,[pos,neg]])
m = len(semester)
p = 1/m
psemester = []
for i in semester:
    pos = ((data[(data.Semester == i) & (data.StudentPerformance == 'Positive')].count()["Semester"])+m*p)/((posCount)+m)
    neg = ((data[(data.Semester == i) & (data.StudentPerformance == 'Negative')].count()["Semester"])+m*p)/((negCount)+m)
    psemester.append([i,[pos,neg]])

testData = pd.read_csv("TestData.csv")

predict = []
for i in testData.itertuples():
    pG = [j[1] for j in pgender if j[0]== i[1]]
    pSt = [j[1] for j in pstage if j[0]== i[2]]
    pSe = [j[1] for j in psemester if j[0]== i[3]]
    #posterior calculation
    posT = Ppos * pG[0][0] * pSt[0][0] * pSe[0][0]
    negT = Pneg * pG[0][1] * pSt[0][1] * pSe[0][1]
    #normalizing
    posN = posT/(posT+negT)
    negN = negT/(posT+negT)
    #print(posN, negN)
    if posN>negN:
        #print('pos')
        predict.append('Positive')
    else:
        #print('neg')
        predict.append('Negative')

testData['PredictedValue'] = predict
testData.to_csv('Predictions.csv')
#confusion matrix values
tp = (testData[(testData.PredictedValue == 'Positive') & (testData.StudentPerformance == 'Positive')].count()["Semester"])
tn = (testData[(testData.PredictedValue == 'Negative') & (testData.StudentPerformance == 'Negative')].count()["Semester"])
fp = (testData[(testData.PredictedValue == 'Positive') & (testData.StudentPerformance == 'Negative')].count()["Semester"])
fn = (testData[(testData.PredictedValue == 'Negative') & (testData.StudentPerformance == 'Positive')].count()["Semester"])

accuracy = (tp+tn)/total
sensitivity= tp/(tp+fn)
specificity = tn/(tn+fp)
print('accuracy: ',accuracy*100)
print('sensitivity',sensitivity*100)
print('specificity', specificity*100)