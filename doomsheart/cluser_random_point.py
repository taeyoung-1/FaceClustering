import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import tensorflow as tf
import math
orbit_arr = []
x_list = [2, 8]
y_list = [2, 8]
sign = [-1, 1]
for i in range(1000):
    orbit_arr.append(((x_list[(random.randrange(0, 2))] + sign[(random.randrange(0, 2))] * pow(random.random(), 2))
                      , (y_list[(random.randrange(0, 2))] + sign[(random.randrange(0, 2))] * pow(random.random(), 2))))

def distance(pt_1, pt_2):
    return math.sqrt((pt_1[0] - pt_2[0])**2 + (pt_1[1] - pt_2[1])**2)

# DB = (1/n)sum(max((sig_i + sig_j)/ d(c_i, c_j)))

def David_Bouldin_index(_centroid, _assignment, _data):
    num_of_clusters = len(_centroid)
    sig_list = []
    for k in range(num_of_clusters):
        k_data = [distance(_data[index_k], _centroid[k]) for index_k in range(len(_assignment)) if
                  _assignment[index_k] == k]
        sig_k = sum(k_data) / len(k_data)
        sig_list.append(sig_k)
    result = 0
    for i in range(num_of_clusters):
        max_value = 0
        for j in range(num_of_clusters):
            if i is j:
                continue
            else:
                sig_i = sig_list[i]
                sig_j = sig_list[j]
                d_ci_cj = distance(_centroid[i], _centroid[j])
                if max_value < (sig_i + sig_j) / d_ci_cj:
                    max_value = (sig_i + sig_j) / d_ci_cj
        result += max_value
    print(result)
    return result

def silhouette(_centroid, _assignment, _points, _i):
    # function a is mean distance between x(i) and data in same cluster.
    def a(__i):
        _cluster = _assignment[__i]
        _x_point = _points[__i]
        _same_cluster_points = [p for a,p in zip(_assignment, _points) if a is _cluster]
        a_val = sum([distance(_s, _x_point) for _s in _same_cluster_points]) / (len(_same_cluster_points) - 1)
        return a_val
    # function b is mean distance between x(i) and data in proximate cluster.
    def b(__i):
        _cluster = _assignment[__i]
        _c_list = [distance(d, _centroid[__i]) for d in _centroid if d is not _centroid[__i]]
        _proximate_cluster = _c_list.index(min(_c_list))

        return

    return b(i) - a(i) / max(a(i), b(i))


def cluster_random_points(k_, points_):
    print("k_ is " + str(k_))
    vectors_ = tf.constant(points_)
    centroids_ = tf.Variable(tf.slice(tf.random_shuffle(vectors_), [0, 0], [k_, -1]))
    expanded_vectors_ = tf.expand_dims(vectors_, 0)
    expanded_centroids_ = tf.expand_dims(centroids_, 1)
    assignments_ = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors_, expanded_centroids_)), 2), 0)
    means_ = tf.concat([tf.reduce_mean(tf.gather(vectors_, tf.reshape(tf.where(tf.equal(assignments_, c)), [1, -1]))
                                      , reduction_indices=[1]) for c in range(k_)], 0)
    update_centroids_ = tf.assign(centroids_, means_)
    init_op_ = tf.global_variables_initializer()
    sess_ = tf.Session()
    sess_.run(init_op_)
    for step_ in range(100):
        _, centroid_values_, assignment_values_ = sess_.run([update_centroids_, centroids_, assignments_])
    return centroid_values_, assignment_values_;


# TODO
# make algorithm that find accurate cluster and evaluate db value so that find appropriate k
k = 3
result_cluster = cluster_random_points(k, orbit_arr)
while True:
    optimized_k_cluster_centroid_and_assignment = cluster_random_points(k, orbit_arr)
    for i in range(10):
        clustering_result = cluster_random_points(k, orbit_arr)
        if David_Bouldin_index(_centroid=optimized_k_cluster_centroid_and_assignment[0],
                               _assignment=optimized_k_cluster_centroid_and_assignment[1],
                               _data=orbit_arr) \
                > \
                David_Bouldin_index(_centroid=clustering_result[0],
                                    _assignment=clustering_result[1],
                                    _data=orbit_arr):
            optimized_k_cluster_centroid_and_assignment = clustering_result
    k += 1
    if David_Bouldin_index(_centroid=result_cluster[0],
                           _assignment=result_cluster[1],
                           _data=orbit_arr) \
            < David_Bouldin_index(_centroid=optimized_k_cluster_centroid_and_assignment[0],
                                  _assignment=optimized_k_cluster_centroid_and_assignment[1],
                                  _data=orbit_arr):
        break
    else:
        result_cluster = optimized_k_cluster_centroid_and_assignment
k -= 2
print("result k is " + str(k))
data = {"x": [], "y": [], "cluster": []}
print(result_cluster[1])
for i in range(len(result_cluster[1])):
    data["x"].append(orbit_arr[i][0])
    data["y"].append(orbit_arr[i][1])
    data["cluster"].append(result_cluster[1][i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()