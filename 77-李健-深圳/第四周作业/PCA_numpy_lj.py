import numpy as np

'''目前能理解PAC降维算法，能调numpy接口实现过程并能理解步骤，暂时还不会手写接口降维方法'''
class PCA():

    def __init__(self, dimension_reduction):
        self.covariance = None
        self.dimension_reduction = dimension_reduction

    def transform(self, X):
        # 去中心化
        X = X - X.mean(axis=0)
        # 协方差矩阵
        self.covariance = np.dot(X.T, X) / X.shape[0]
        # 得到特征值和特征向量
        eig_var, eig_vector = np.linalg.eig(self.covariance)
        # 降序
        index = np.argsort(-eig_var)
        print(eig_vector)
        components = eig_vector[:, index[:self.dimension_reduction]]
        print(components)
        return np.dot(X, components)


pca = PCA(dimension_reduction=2)
X = np.array(
    [[3, 4, 3, 23], [-12, 66, 88, 67], [53, 81, 40, -72], [1, 3, 3, 1], [92, 180, 6, -33],
     [13, 35, -83, 22]])  # 创建测试数据，维度为4
newX = pca.transform(X)
print(newX)  # 输出降维后的数据
