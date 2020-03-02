#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
##########################################################################
# File Name: LDA.py
# Author: stubborn vegeta
# Created Time: 2020年02月28日 星期五 14时35分31秒
##########################################################################
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def lda(data,target,n_dim):
    """
    :data: data
    :target: label
    :n_dim: dim after LDA
    """
    label = np.unique(target)
    Nlabel = len(label)-1
    if n_dim > Nlabel:
        return print("n_dim should <=",Nlabel)
    else:
        u = {}              #sub_data
        Sw = np.zeros((data.shape[1],data.shape[1]))
        for i in label:
            u[i]    = data[target == i]
            u[i]    = np.mean(u[i],axis=0)
            u[i]    = np.mat(u[i])
            sub_Sw  = (u[i].T) * u[i]
            Sw     += sub_Sw

        Sb = np.zeros((data.shape[1],data.shape[1]))
        u["all"] = np.mean(list(u.values()),axis=0)

        for i in label:
            sub_Sb = (u[i]-u["all"]).dot((u[i]-u["all"]).T)
            Sb    += sub_Sb

        S = np.linalg.inv(Sw).dot(Sb)
        eigVals,eigVects = np.linalg.eig(S)
        eigValInd = np.argsort(-eigVals)
        eigValInd = eigValInd[:n_dim]
        w = eigVects[:,eigValInd]
        data_ndim = np.dot(data,w)

        return data_ndim


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    data = lda(X,Y,2)
    data_2 = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, Y)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_LDA")
    plt.scatter(data[:,0],data[:,1],c=Y)
    plt.subplot(122)
    plt.title("sklearn_LDA")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    plt.savefig("LDA.png")
    plt.show()
    # print(X.shape,Y.shape)
