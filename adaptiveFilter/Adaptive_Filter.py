#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   Adaptive_Filter.py
@Time    :   2020/11/10 10:53:48
@Author  :   xingjiezhen
@Version :   1.0
@Contact :   xingjiezhen@buaa.edu.cn
@Desc    :   Description
'''
import numpy as np
import LMS


def addNoise(x, sigma):
    noise = np.random.normal(0, sigma, size = [1, len(x)])
    x_noise = x + noise
    return x_noise