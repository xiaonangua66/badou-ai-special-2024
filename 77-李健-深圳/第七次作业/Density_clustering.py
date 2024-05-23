'''密度聚类'''
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import datasets


def density_clustering():
    #加载鸢尾花数据
    iris = datasets.load_iris()
    X = iris.data[:, :4]
    dbscan = DBSCAN(eps=0.4, min_samples=9)
    dbscan.fit(X)

    label_pred = dbscan.labels_
    print(label_pred)
    # 开始绘图
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    x2 = X[label_pred == 2]
    plt.scatter(x0[:, 0], x0[:, 1], c='red', marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c='black', marker='*', label='label1')
    plt.scatter(x2[:, 0], x2[:, 1], c='blue', marker='+', label='label2')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=2)
    plt.show()


#调用方法
density_clustering()