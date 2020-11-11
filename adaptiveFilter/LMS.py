#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   LMS.py
@Time    :   2020/11/09 16:05:24
@Author  :   xingjiezhen
@Version :   1.0
@Contact :   xingjiezhen@buaa.edu.cn
@Desc    :   Description
'''
import numpy as np
import matplotlib.pyplot as plt

class LMSMethod(object):
    def doLMS(xn,dn,M,mu):
        itr = len(xn)
        en = np.zeros((itr, 1))
        W = np.zeros((M, itr))
        for k in range(M, itr):
            x = xn[k:k-M+1:-1]
            y = np.dot(W[:, k-1], x)
            en[k] = dn[k] - y
            W[:, k] = np.add(W[:, k-1] + 2*mu*en(k)*x)
        
        yn = float("inf")*np.ones(len(xn))
        for k in range(len(xn))[M-1:len(xn)]:
            x = xn[k:k-M+1:-1]
            yn[k] = np.dot(W[:, -1], x)
        return yn,en

