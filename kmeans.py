import numpy as np
from numpy.polynomial import laguerre


class Kmeans:

    def __init__(self, k,max_iter, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, type="euclidean"):
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centers = X[indices]
        for iteration in range(self.max_iter):
            # 利用广播机制
            if type == "euclidean":
                distances = np.linalg.norm(X[:, None,:] - self.centers[None,:,:], axis = 2) # in the dimension of feature
                labels = np.argmin(distances, axis =1)
                # 更新聚类中心
                new_centers = np.zeros_like(self.centers)
                converged = True
                for i in range(self.k):
                    cluster_points = X[labels == i]  # 对每一个分类的样本求平均值
                    if len(cluster_points) > 0:
                        new_centers[i] = cluster_points.mean(axis=0)
                    else:
                        # 空簇：重新随机选一个点（或保留旧中心）
                        new_centers[i] = self.centers[i]  # 或者 X[np.random.choice(n_samples)]
                    
                    # 检查是否移动过大
                    if np.linalg.norm(new_centers[i] - self.centers[i]) > self.tol:
                        converged = False

                self.centers = new_centers
                self.labels = labels
                if converged:
                    print(f"Converged at iteration {iteration + 1}")
                    break
                self.inertia_ = np.sum(np.argmin(distances, axis =1)**2)
                return self

    def predict(self, X):
        """预测新数据的标签"""
        diff = X[:, None, :] - self.centers[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return np.argmin(distances, axis=1)

                
# 生成数据
X = np.random.rand(100, 2)

# 训练
kmeans = Kmeans(k=3, max_iter=100)
kmeans.fit(X)

print("Cluster centers:\n", kmeans.centers)
print("Labels:", kmeans.labels[:10])  # 前10个样本的标签
print("Inertia:", kmeans.inertia_)




        