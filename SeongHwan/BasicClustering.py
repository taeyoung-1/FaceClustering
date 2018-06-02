import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import tensorflow as tf

import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

orbit_arr = []
x_list = [2, 4]
y_list = [2, 4]
sign = [-1, 1]
for i in range(1000):
    orbit_arr.append(((x_list[(random.randrange(0, 2))] + sign[(random.randrange(0, 2))] * pow(random.random(), 2))
                      , (y_list[(random.randrange(0, 2))] + sign[(random.randrange(0, 2))] * pow(random.random(), 2))))

# df = pd.DataFrame({"x": [v[0] for v in orbit_arr], "y": [v[1] for v in orbit_arr]})
#
# sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
# # plt.show()

##클러스터 계수 구하기(실루엣 기법)
def plotSilhouette(X, y_km):
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric = 'euclidean')
    y_ax_lower, y_ax_upper = 0,0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i/n_clusters)

        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height = 1.0, edgecolor ='none', color = color)
        yticks.append((y_ax_upper + y_ax_lower)/2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color = 'red', linestyle = '--')
    plt.yticks(yticks, cluster_labels+1)
    plt.ylabel('클러스터')
    plt.xlabel('실루엣 계수')
    plt.show()

X, y = make_blobs(n_samples = 150, n_features = 2, centers = 3, cluster_std = 0.5, shuffle = True, random_state = 0)
km = KMeans(n_clusters = 3, random_state = 0) #이 부분에서 n_cluster에 1,2,3,... 넣어서 모두 비슷하게 나오는 값이
                                             #K-Mean 값으로 적당함
y_km = km.fit_predict(X)
plotSilhouette(X, y_km)


vectors = tf.constant(orbit_arr)
k = km
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2), 0)

means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1]))
                                  , reduction_indices=[1]) for c in range(k)], 0)

update_centroids = tf.assign(centroids, means)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for step in range(100):
   _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(orbit_arr[i][0])
    data["y"].append(orbit_arr[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()
