
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# sklearn.cluster.KMeans.fit

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import numpy as np

# Generate isotropic Gaussian blobs for clustering.
# scikit中的make_blobs方法常被用来生成聚类算法的测试数据，直观地说，make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。

X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5,shuffle=True, random_state=0)
print(type(X))


fig = plt.figure(figsize=(20,5))
#oriFig = fig.add_subplot(1,3,1)
plt.subplot(1,3,1)
plt.scatter(X[:,0],X[:,1],c="red",marker="o",s=10)
plt.xlabel("x")
plt.ylabel("y")

km = KMeans(n_clusters=3,init="random",n_init=10,max_iter=300)
y_km = km.fit_predict(X)

print(type(y_km))
print(y_km)


#resultFig = fig.add_subplot(1,3,2)
plt.subplot(1,3,2)
plt.scatter(X[y_km==0,0],X[y_km==0,1],s=10,c="lightgreen",marker="s", label="cluster 1")
plt.scatter(X[y_km==1,0],X[y_km==1,1],s=10,c="orange",marker="o", label="cluster 2")
plt.scatter(X[y_km==2,0],X[y_km==2,1],s=10,c="lightblue",marker="v", label="cluster 3")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=10,c="red",marker="*", label="centroids")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")




distortions = []
for i in range(1,11):
    km = KMeans(n_clusters = i, init="k-means++",n_init=10,max_iter=300,random_state=0)
    # 	Compute k-means clustering.
    km.fit(X)
    # Sum of distances of samples to their closest cluster center. e.g. km.inertia_
    distortions.append(km.inertia_)

# print(distortions)
#elbowFig = fig.add_subplot(1,3,3)
plt.subplot(1,3,3)
plt.plot(range(1,11),distortions,marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")


plt.grid()
plt.show()



## Recall ****************
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.arange(7)
print(names)
print(data)
print("Bob data =", data[names == 'Bob'])


x1 = np.array(np.arange(7))
x2 = np.array(np.arange(7))
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

alist = []
i = 0;
for i in range(len(x1)):
    alist.append([x1[i],x2[i]])

data = np.array(alist)


print(data)


print("Bob data =", data[names == 'Bob'])