#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   Tool_Service.py
@Time    :   2020/11/03 16:47:28
@Author  :   xingjiezhen
@Version :   1.0
@Contact :   xingjiezhen@buaa.edu.cn
@Desc    :   Description
'''
import matplotlib.pyplot as graph
import numpy as np
from numpy import fft
import math
import cv2

class Wiener_Tool():
    # 仿真运动模糊
    def motion_process(image_size, motion_angle):
        PSF = np.zeros(image_size)
        print(image_size)
        center_position = (image_size[0] - 1) / 2
        print(center_position)
    
        slope_tan = math.tan(motion_angle * math.pi / 180)
        slope_cot = 1 / slope_tan
        if slope_tan <= 1:
            for i in range(15):
                offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
                PSF[int(center_position + offset), int(center_position - offset)] = 1
            return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
        else:
            for i in range(15):
                offset = round(i * slope_cot)
                PSF[int(center_position - offset), int(center_position + offset)] = 1
            return PSF / PSF.sum()


    # 对图片进行运动模糊
    def make_blurred(input, PSF, eps):
        input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
        PSF_fft = fft.fft2(PSF) + eps
        blurred = fft.ifft2(input_fft * PSF_fft)
        blurred = np.abs(fft.fftshift(blurred))
        return blurred


    def inverse(input, PSF, eps):  # 逆滤波
        input_fft = fft.fft2(input)
        PSF_fft = fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
        result = fft.ifft2(input_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
        result = np.abs(fft.fftshift(result))
        return result


    def wiener(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
        input_fft = fft.fft2(input)
        PSF_fft = fft.fft2(PSF) + eps
        PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
        result = fft.ifft2(input_fft * PSF_fft_1)
        result = np.abs(fft.fftshift(result))
        return result
