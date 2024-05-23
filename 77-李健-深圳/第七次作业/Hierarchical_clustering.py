'''层次聚类'''
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt


def hierarchical_clustering():
    X = [[2, 3], [2, 1], [4, 2], [3, 2], [1, 4]]
    Z = linkage(X, 'ward')
    f = fcluster(Z, 4, 'distance')
    fig = plt.figure(figsize=(5, 3))
    dn = dendrogram(Z)
    plt.show()


hierarchical_clustering()
