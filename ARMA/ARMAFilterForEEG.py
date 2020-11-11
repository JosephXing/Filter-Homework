#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   ARMAFilter.py
@Time    :   2020/11/11 17:37:00
@Author  :   xingjiezhen
@Version :   1.0
@Contact :   xingjiezhen@buaa.edu.cn
@Desc    :   Description
'''
import pyflux as pf
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def doARMA(data, ar, ma):
    family = pf.Normal()
    model = pf.ARIMA(data=data, ar=ar, ma=ma, target='sunspot.year', family=family)
    x = model.fit("MLE")
    x.summary()
    # model.plot_fit(figsize=(15,10))
    model.plot_predict_is(h=500, figsize=(15,5))
    # model.plot_predict(h=20,past_values=20,figsize=(15,5))
    res = model.predict_is(h=500)
    return res


def main():
    data = sio.loadmat('ARMA\mydata.mat')
    dataMat = data['d']

    rawData = dataMat[1, :]
    trainData = rawData[0:1000]
    testData = rawData[1000:1500]
    res = doARMA(trainData, 5, 5)
    predict = np.array(res)
    # print(testData.shape)
    plt.figure(figsize=(15,5))
    plt.plot(testData, color='b')
    plt.plot(predict, color='r')

    plt.show()


if __name__ == "__main__":
    main()