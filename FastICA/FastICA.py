#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
##########################################################################
# File Name: FastICA.py
# Author: stubborn vegeta
# Created Time: 2020年02月23日 星期日 13时55分28秒
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA


def ICA(S,N,Maxcount=10000,precision=0.00001):
    """
    :S        : 数据
    :N        : 独立成分个数
    :Maxcount : 最大迭代次数
    :precision: 精度
    """
    for i in range(N):
        S[i,:] = S[i,:] - np.mean(S,axis=1)[i]
    Cx = np.cov(S)
    value,eigvector = np.linalg.eig(Cx)
    print(value, eigvector)
    val = value**(-1/2)*np.eye(N)
    white = np.dot(val, eigvector.T)
    Z = np.dot(white, S)
    W = 0.5*np.ones([N,N])

    for n in range(N):
        count=0
        WP=W[:,n].reshape(N,1) #初始化
        LastWP=np.zeros(N).reshape(N,1) # 列向量;LastWP=zeros(m,1);
        while np.linalg.norm(WP-LastWP,1)>precision:
            count=count+1
            LastWP=np.copy(WP)    #  上次迭代的值
            gx=np.tanh(LastWP.T.dot(Z))  # 行向量

            for i in range(N):
                tm1=np.mean( Z[i,:]*gx )
                tm2=np.mean(1-gx**2)*LastWP[i]
                WP[i]=tm1 - tm2
            #print(" wp :", WP.T )
            WPP=np.zeros(N) #一维0向量   史密特正交
            for j in range(n):
                WPP=WPP+  WP.T.dot(W[:,j])* W[:,j]
            WP.shape=1,N
            WP=WP-WPP
            WP.shape=N,1
            WP=WP/(np.linalg.norm(WP))
            if(count == Maxcount):
                # print("reach Maxcount，exit loop",np.linalg.norm(WP-LastWP,1))
                break
        W[:,n]=WP.reshape(N,)
    SZ=W.T.dot(Z)

    return SZ.T


if __name__ == '__main__':
    simple=200
    x = np.arange(simple)
    s1 = np.sin(0.01*np.pi*x)            #(simple,1)
    s2 = np.array(20*(5*[2]+5*[-2]))     #(simple,1)
    S = s1*s2
    S = np.array([s1,s2])
    R,C = S.shape
    ran=np.random.random([R,R])  #随机矩阵
    S=ran.dot(S) #混合信号
    print(S.shape)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("mixed signal 1")
    plt.plot(S[0])
    plt.subplot(122)
    plt.title("mixed signal 2")
    plt.plot(S[1])
    plt.savefig("MixedSignal")
    plt.show()
    SZ = ICA(S,R)
    ica=FastICA(n_components=2)
    SZ_1 = ica.fit_transform(S.T)
    plt.figure(figsize=(8,4))
    # plt.plot(SZ.T[:,0])
    plt.subplot(121)
    plt.title("my_ICA")
    plt.plot(SZ[:,1])
    plt.plot(SZ[:,0])
    plt.subplot(122)
    plt.title("sklearn_ICA")
    plt.plot(SZ_1[:,1])
    plt.plot(SZ_1[:,0])
    plt.savefig("FastICA")
    plt.show()
