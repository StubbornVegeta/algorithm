#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
##########################################################################
# File Name: PCA.py
# Author: stubborn vegeta
# Created Time: 2019年12月26日 星期四 20时08分46秒
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def my_PCA(dataSet, n_dim):
    """
    :dataSet : 数据集
    :n_dim   : 要降低的维度
    """
    meanValues  =  np.mean(dataSet, axis=1)
    N           =  dataSet.shape[0]

    for i in range(N):
        dataSet[i] = dataSet[i] - meanValues[i]

    covMat               =   np.cov(dataSet, rowvar=True)
    eigValues, eigVects  =   np.linalg.eig(np.mat(covMat))
    eigValIndex          =   np.argsort(-eigValues)
    eigValIndex          =   eigValIndex[:n_dim]
    redEigVects          =   eigVects[:,eigValIndex]
    dataSet              =   np.array(dataSet)
    redEigVects          =   np.array(redEigVects)
    newDataSet           =   dataSet.T.dot(redEigVects)
    pcaData              =   (newDataSet.dot(redEigVects.T))
    return pcaData.T


if __name__ == '__main__':
    iris         =   load_iris()
    X            =   iris.data
    Y            =   iris.target
    sklearn_pca  =   PCA(n_components=2)
    data_2       =   sklearn_pca.fit_transform(X)
    data_1       =   my_PCA(X, 2)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_PCA")
    plt.scatter(data_1[:,0], data_1[:,1], c=Y)
    plt.subplot(122)
    plt.title("sklearn_PCA")
    plt.scatter(data_2[:,0], data_2[:,1], c=Y)
    plt.savefig("PCA.png")
    plt.show()
