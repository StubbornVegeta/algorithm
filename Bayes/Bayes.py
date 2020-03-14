#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
##########################################################################
# File Name: Bayes.py
# Author: stubborn vegeta
# Created Time: 2020年03月13日 星期五 21时10分51秒
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def BayesClassfier(train, test):
    """
    :贝叶斯分类器（高斯贝叶斯分类器）
    :train:训练集
    :test :测试集
    """
    labels = np.unique(train[:,-1])
    mu     = []                         # 均值
    sigma  = []                         # 方差
    # 求训练集中每一类数据对应的方差和均值
    for i in labels:
        tmp = train[train[:,-1]==i][:,:-1]
        tmpMu = tmp.mean(axis=0)
        mu.append(tmpMu)
        sigma.append((sum((tmp-tmpMu)**2)/tmp.shape[0]))
    mu = np.array(mu)
    sigma = np.array(sigma)

    cla = []
    # 对测试集的每一个样本进行预测
    for i in range(test.shape[0]):
        tmp = test[i,:-1].reshape(1,-1)
        # 将均值和方差带入高斯贝叶斯的公式
        pi = np.exp(-1*(tmp-mu)**2/(sigma*2))/(np.sqrt(2*np.pi*sigma))
        # 按照朴素贝叶斯的规则，将各部分的概率相乘
        Pi = np.ones((labels.shape[0],1))
        for k in range(pi.shape[1]):
            Pi = Pi*pi[:,k].reshape(-1,1)
        cla.append(np.argmax(Pi))
    return cla

def accuracy(cla,target):
    """
    :计算预测准确率
    :cla   :你的预测结果
    :target:实际结果
    """
    cnt = 0
    # count = 0
    count = target.shape[0]
    acc = []
    for i,label in enumerate(cla):
        if label-target[i]:
            pass
        else:
            cnt += 1
        # count += 1
        acc.append(cnt/count)
    print(acc[-1])
    return acc


if __name__ == '__main__':
    iris = load_iris()
    # 分割数据集，和测试集
    train_data, test_data, train_target, test_target = train_test_split(iris.data,iris.target)
    train_target = train_target.reshape(-1,1)
    test_target = test_target.reshape(-1,1)
    train = np.hstack((train_data,train_target))
    test = np.hstack((test_data,test_target))

    cla = BayesClassfier(train,test)
    myAcc = accuracy(cla,test_target)

    # sklearn GaussianNB
    gnb = GaussianNB()
    gnb.fit(train_data,train_target)
    y_predict = gnb.predict(test_data)
    sklearnAcc = accuracy(y_predict,test_target)

    x = [i for i in range(test.shape[0])]

    plt.figure()
    plt.plot(x,myAcc,color='red',LineStyle='--',LineWidth=4,label='my bayes',alpha=0.7)
    plt.plot(x,sklearnAcc,color='green',LineStyle='-.',LineWidth=2,label='sklearn bayes')
    plt.legend(loc='best')
    plt.savefig('bayes')
    plt.show()
