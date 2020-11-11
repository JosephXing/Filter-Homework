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
    # x.summary()
    # model.plot_fit(figsize=(15,10))
    model.plot_predict_is(h=500, figsize=(15,5))
    # model.plot_predict(h=20,past_values=20,figsize=(15,5))
    res = model.predict_is(h=500)
    return res

def genateSignal():
    t = np.linspace(0,10,2500)
    y = 10 * (np.sin(2*np.pi*t)+np.sin(4*np.pi*t))
    noise = np.random.randn(2500)
    y_noise = y + noise
    return y_noise

def mse(signal1, signal2):
    res = 0
    for i in range(0, len(signal1)):
        res += (signal1[i] - signal2[i])**2
    return res


def main():
    data = genateSignal()
    trainData = data[0:1000]
    testData = data[1000:1500]
    plt.figure(figsize=(15,5))
    res = doARMA(trainData, 5, 0)
    predict = np.array(res)
    # print(testData.shape)
    plt.figure(figsize=(15,5))
    plt.plot(testData, label='test data', color='b')
    plt.plot(predict, label='predict data', color='r')
    plt.show()
    err = mse(testData, predict)
    print('error is: ' + str(err))

if __name__ == "__main__":
    main()