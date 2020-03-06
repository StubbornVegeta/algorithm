#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
##########################################################################
# File Name: Tsne.py
# Author: stubborn vegeta
# Created Time: 2020年03月04日 星期三 21时16分12秒
##########################################################################
from sklearn.datasets import load_digits,load_iris
from sklearn import manifold
from sklearn.decomposition import PCA,FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


if __name__ == '__main__':
    digit = load_digits()
    X = digit.data
    Y = digit.target

    # iris = load_iris()
    # X = iris.data
    # Y = iris.target

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X,Y)

    ica = FastICA(n_components=2)
    X_ica = ica.fit_transform(X)

    tsne = manifold.TSNE(n_components=2,early_exaggeration=2, init='pca', random_state=1)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8,8))
    plt.subplot(221)
    plt.title("sklearn_PCA")
    plt.scatter(X_pca[:,0],X_pca[:,1],c=Y)

    plt.subplot(222)
    plt.title("sklearn_LDA")
    plt.scatter(X_lda[:,0],X_lda[:,1],c=Y)

    plt.subplot(223)
    plt.title("sklearn_ICA")
    plt.scatter(X_ica[:,0],X_ica[:,1],c=Y)

    plt.subplot(224)
    plt.title("sklearn_T SNE")
    plt.scatter(X_tsne[:,0],X_tsne[:,1],c=Y)
    plt.savefig("TSNEbest.png")
    plt.show()

    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(Y[i]), color=plt.cm.Set1(Y[i]),
                 # fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

