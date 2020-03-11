#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
##########################################################################
# File Name: csp.py
# Author: stubborn vegeta
# Created Time: 2020年03月11日 星期三 11时14分30秒
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from mpl_toolkits.mplot3d import Axes3D

def csp(X1,X2):
    X1     =  X1 - np.mean(X1, axis=2, keepdims=True)
    X2     =  X2 - np.mean(X2, axis=2, keepdims=True)
    count1 =  X1.shape[0]
    count2 =  X2.shape[0]
    C1     =  []
    C2     =  []

    # C = sum[(X*X.T)/trace(X*X.T)]
    for i in range(count1):
        tmp = np.dot(X1[i,:,:], X1[i,:,:].T)
        C1.append(tmp/np.trace(tmp))
    for i in range(count2):
        tmp = np.dot(X2[i,:,:], X2[i,:,:].T)
        C2.append(tmp/np.trace(tmp))

    C1 = np.array(C1)
    C2 = np.array(C2)

    mean_C1 = C1.mean(axis=0)
    mean_C2 = C2.mean(axis=0)

    mean_C = np.linalg.inv(mean_C2).dot(mean_C1)
    # max((W.T*mean_C1*W)/(W.T*mean_C2*W))
    eigValFilter, eigVectFilter = np.linalg.eig(mean_C)
    # eigValFilter, eigVectFilter = linalg.eigh(mean_C1, mean_C2)
    return eigValFilter, eigVectFilter


if __name__ == '__main__':
    Nc       = 3            # 导联数
    Simpling = 1024         # 采样点数
    count    = 20           # 试验次数
    dataset1 = np.log(np.random.rand(count, Nc, Simpling))
    dataset2 = np.random.rand(count, Nc, Simpling)
    eigValFilter, eigVectFilter = csp(dataset1,dataset2)


    # meanX1 = np.mean(dataset1, axis=2).reshape((Nc,count,1))
    # meanX2 = np.mean(dataset2, axis=2).reshape((Nc,count,1))
    # eigValFilter = eigValFilter.reshape((Nc,1,1))
    # # data1 = np.dot(eigVectFilter, dataset1)*(eigValFilter) + meanX1
    # # data2 = np.dot(eigVectFilter, dataset2)*(eigValFilter) + meanX2
    # data1 = np.dot(eigVectFilter, dataset1) + meanX1
    # data2 = np.dot(eigVectFilter, dataset2) + meanX2
    # # # data1 = np.dot(eigVectFilter, dataset1)
    # # # data2 = np.dot(eigVectFilter, dataset2)

    # plt.figure(figsize=(9,6))
    # plt.subplot(231)
    # plt.scatter(dataset1[0,0,:],dataset1[0,1,:])
    # plt.scatter(dataset2[0,0,:],dataset2[0,1,:],c="red")
    # plt.subplot(232)
    # plt.scatter(dataset1[0,0,:],dataset1[0,2,:])
    # plt.scatter(dataset2[0,0,:],dataset2[0,2,:],c="red")
    # plt.subplot(233)
    # plt.scatter(dataset1[0,1,:],dataset1[0,2,:])
    # plt.scatter(dataset2[0,1,:],dataset2[0,2,:],c="red")
    # plt.subplot(234)
    # plt.scatter(data1[0,0,:],data1[0,1,:])
    # plt.scatter(data2[0,0,:],data2[0,1,:],c="red")
    # plt.subplot(235)
    # plt.scatter(data1[0,0,:],data1[0,2,:])
    # plt.scatter(data2[0,0,:],data2[0,2,:],c="red")
    # plt.subplot(236)
    # plt.scatter(data1[0,1,:],data1[0,2,:])
    # plt.scatter(data2[0,1,:],data2[0,2,:],c="red")
    # plt.show()
