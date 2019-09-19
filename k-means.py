# https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN



train = pd.read_csv("/Users/qingli/Desktop/data/optdigits.tra", header = None)

test = pd.read_csv("/Users/qingli/Desktop/data/optdigits.tes", header = None)

# print(train.tail(3))
# print(np.arange(9))


# traning data
X_train = train[np.arange(64)]
#X_train = train.iloc[:,:64]
y_train = train[64]
# print(X_train.tail(3))
#print(m_train.tail(3))


#testing Data
X_test = test[np.arange(64)]
y_test = test[64]

#construct the model
km = KMeans(n_clusters=10,init="k-means++",n_init=10,max_iter=300,random_state=0)
#train the model
km.fit(X_train)
# using well-trained model to predict
y_pred = km.predict(X_test)
# evaluate the predictions
score = metrics.adjusted_rand_score(y_test,y_pred)
print(score)

# silhouette coefficient for sample x_i, 1. cohension a_i 2. Separation b_i(min), sc= (b_i-a_i)/max(b_i-a_i)

fig = plt.figure(figsize=(15,5))
plt.subplot(3, 2, 1)
#
# x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
# x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
# data = np.array([x1,x2]).reshape(14,2) ## np.array(zip(x1,x2)).reashape(len(x1),2)
# X = pd.DataFrame(data)



x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])


# alist = []
# i = 0;
# for i in range(len(x1)):
#     alist.append([x1[i],x2[i]])
#
# data = np.array(alist)
#
# X = pd.DataFrame(data)



# x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
# x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
# X = np.array(zip(x1, x2)).reshape(len(x1), 2)

print("\n",X)
# print(len(x1),len(x2))
# X = np.array(zip(x1, x2)).reshape(len(x1), 2)

# plt.xlim([0,10])
# plt.ylim([0,10])
plt.title("instance")
plt.scatter(x1,x2)
colors = ["b","g","r","c","m","y","k","b"]
markers = ["o","s","D","v","^","p","*","+"]

clusters = [2,3,4,5,8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter +=1
    plt.subplot(3,2,subplot_counter)
    km = KMeans(n_clusters=t,init="k-means++",n_init=10,max_iter=300)

    km.fit(X)
    print(km.labels_)
    for i, l in enumerate(km.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l])
    # plt.xlim([0, 10])
    # plt.ylim([0, 10])
    sc_score = silhouette_score(X, km.labels_, metric='euclidean')
    sc_scores.append(sc_score)

    # 绘制轮廓系数与不同类簇数量的直观显示图。
    plt.title('K = %s, SC= %0.03f' % (t, sc_score),fontsize = "small")

# 绘制轮廓系数与不同类簇数量的关系曲线。
plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('SC Score')



distortions = []
for i in clusters:
    km = KMeans(n_clusters = i, init="k-means++",n_init=10,max_iter=300,random_state=0)

    # 	Compute k-means clustering.
    km.fit(X)
    # Sum of distances of samples to their closest cluster center. e.g. km.inertia_
    distortions.append(km.inertia_)


print(distortions)
plt.figure()
plt.plot(clusters,distortions,marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")


# DBSCAN
plt.figure()
km = DBSCAN(eps=0.2, min_samples=2, metric='euclidean')
km.fit(X)
for i, l in enumerate(km.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l])

plt.show()