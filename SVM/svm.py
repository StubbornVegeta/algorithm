#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
##########################################################################
# File Name: svm.py
# Author: stubborn vegeta
# Created Time: 2020年03月16日 星期一 20时52分48秒
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data(filename):
    data = []
    label = []
    with open(filename,"r") as f:
        for line in f:
            x, y, cla = line.split()
            data.append([float(x),float(y)])
            label.append(float(cla))
    return np.array(data), np.array(label)


class SMO():
    def __init__(self, data, label, C, iterations=40):
        """
        :data      :数据
        :label     :标签
        :C         :软间隔常数
        :iterations:最大迭代次数
        """
        self.data       = data
        self.label      = label
        self.C          = C
        self.iterations = iterations
        self.m,_        = self.data.shape
        self.alpha      = np.zeros(self.m)        # 一维向量
        self.b          = 0
        self.cnt        = 0

    def f(self, x):
        """
        预测值
        f = w.T*x+b
          = sum(alpha_i*y_i*x_i.T*x)+b
        """
        fx = (self.alpha*self.label).dot(x.dot(self.data.T)) + self.b
        return fx

    def solution(self):
        while self.cnt < self.iterations:
            for i in range(self.m):
                jlist = [x for x in range(i)] + [x for x in range(i+1,self.m)]
                j = random.choice(jlist)
                fx_i = self.f(self.data[i])
                E_i = fx_i - self.label[i]
                fx_j = self.f(self.data[j])
                E_j = fx_j - self.label[j]
                K_ii = self.data[i].T.dot(self.data[i])
                K_jj = self.data[j].T.dot(self.data[j])
                K_ij = self.data[i].T.dot(self.data[j])
                eta = K_ii + K_jj - 2*K_ij

                alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                alpha_j_new = alpha_j_old + self.label[j]*(E_i - E_j)/eta

                if self.label[i] != self.label[j]:
                    L = max(0, alpha_j_old-alpha_i_old)
                    H = min(self.C, self.C+alpha_j_old-alpha_i_old)
                else:
                    L = max(0, alpha_j_old+alpha_i_old-self.C)
                    H = min(self.C, alpha_j_old+alpha_i_old)

                # 将alpha值约束到(L,H)
                if alpha_j_new < L:
                    alpha_j_new = L
                elif alpha_j_new > H:
                    alpha_j_new = H

                alpha_i_new = alpha_i_old + self.label[i]*(self.label[j])*(alpha_j_old-alpha_j_new)

                if abs(alpha_j_new-alpha_j_old) < 0.00001:
                    continue

                self.alpha[i], self.alpha[j] = alpha_i_new, alpha_j_new

                b_i = -E_i - self.label[i]*K_ii*(alpha_i_new-alpha_i_old) - self.label[j]*K_ij*(alpha_j_new-alpha_j_old) + self.b
                b_j = -E_j - self.label[i]*K_ij*(alpha_i_new-alpha_i_old) - self.label[j]*K_jj*(alpha_j_new-alpha_j_old) + self.b

                if 0 < alpha_i_new < self.C:
                    self.b = b_i
                elif 0 < alpha_j_new < self.C:
                    self.b = b_j
                else:
                    self.b = (b_i + b_j)/2

            self.cnt += 1

        return self.alpha, self.b


if __name__ == '__main__':
    data,labels =load_data('./testSet.txt')
    smo = SMO(data,labels,0.6,100)

    alphas,b = smo.solution()
    print(alphas,b)
    w = (smo.alpha*smo.label).reshape(1,-1).dot(smo.data)
    y1 = (-smo.b - w[0,0]*0)/w[0,1]
    y2 = (-smo.b - w[0,0]*10)/w[0,1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0],data[:,1],c=labels)
    ax.plot([0,10], [y1,y2])

    # 绘制支持向量
    # for i, alpha in enumerate(alphas):
        # if abs(alpha) > 1e-3:
            # x, y = data[i]
            # ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       # linewidth=1.5, edgecolor='#AB3319')
    plt.savefig("svm")
    plt.show()


