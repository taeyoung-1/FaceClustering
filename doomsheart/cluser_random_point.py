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
            if i == j:
                continue
            else:
                sig_i = sig_list[i]
                sig_j = sig_list[j]
                d_ci_cj = distance(_centroid[i], _centroid[j])
                if max_value < (sig_i + sig_j) / d_ci_cj:
                    max_value = (sig_i + sig_j) / d_ci_cj
        result += max_value
    # print(result)
    return result

def silhouette(_centroid, _assignment, _points, _i):
    # function a is mean distance between x(i) and data in same cluster.
    def a(__i):
        _curr_cluster = _assignment[__i]
        _x_point = _points[__i]
        _same_cluster_points = [p for a,p in zip(_assignment, _points) if a == _curr_cluster]
        try:
            a_val = sum([distance(_s, _x_point) for _s in _same_cluster_points]) / (len(_same_cluster_points) - 1)
        except:
            print("======================a_val error======================")
            print(_centroid, _assignment, _points, _i)
            a_val = 1

        # print("a_val is " + str(a_val))
        return a_val
    # function b is mean distance between x(i) and data in proximate cluster.
    def b(__i):
        _curr_cluster = _assignment[__i]
        _x_point = _points[__i]
        _dis_list = [distance(_centroid[_curr_cluster], _c) for _c in _centroid]
        # print(_dis_list)
        _proximate_cluster = -1
        _min_dis = sum(_dis_list)
        for i in range(len(_dis_list)):
            if (_dis_list[i] != 0) and (_min_dis > _dis_list[i]):
                _proximate_cluster = i
                _min_dis = _dis_list[i]
        if _proximate_cluster == -1:
            return 0
        # print(_curr_cluster, _proximate_cluster)
        _proximate_cluster_points = [p for a,p in zip(_assignment, _points) if a == _proximate_cluster]
        try :
            b_val = sum([distance(_s, _x_point) for _s in _proximate_cluster_points]) / (len(_proximate_cluster_points) - 1)
        except:
            print("======================a_val error======================")
            print(_centroid, _assignment, _points, _i)
            b_val = 1
        return b_val
    a_i = a(_i)
    b_i = b(_i)
    sil = (b_i - a_i) / max(a_i, b_i)
    # print(a_i,b_i, max(a_i, b_i), sil)
    
    # print("silhouette is " + str(sil))
    return sil


def cluster_random_points(k_, points_):
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
    for _ in range(100):
        _, centroid_values_, assignment_values_ = sess_.run([update_centroids_, centroids_, assignments_])
    return centroid_values_, assignment_values_

def evaluate_cluster(_centroid, _assignment, _data):
    result = sum([silhouette(_centroid,_assignment,_data,i) for i in range(len(_assignment))]) / len(_assignment)
    # print(result)
    return result

# # TODO
# # make algorithm that find accurate cluster and evaluate db value so that find appropriate k
# k = 3
# result_cluster = cluster_random_points(k, orbit_arr)
# while True:
#     optimized_k_cluster_centroid_and_assignment = cluster_random_points(k, orbit_arr)
#     for i in range(10):
#         clustering_result = cluster_random_points(k, orbit_arr)
#         if David_Bouldin_index(_centroid=optimized_k_cluster_centroid_and_assignment[0],
#                                _assignment=optimized_k_cluster_centroid_and_assignment[1],
#                                _data=orbit_arr) \
#                 > \
#                 David_Bouldin_index(_centroid=clustering_result[0],
#                                     _assignment=clustering_result[1],
#                                     _data=orbit_arr):
#             optimized_k_cluster_centroid_and_assignment = clustering_result
#     k += 1
#     if David_Bouldin_index(_centroid=result_cluster[0],
#                            _assignment=result_cluster[1],
#                            _data=orbit_arr) \
#             < David_Bouldin_index(_centroid=optimized_k_cluster_centroid_and_assignment[0],
#                                   _assignment=optimized_k_cluster_centroid_and_assignment[1],
#                                   _data=orbit_arr):
#         break
#     else:
#         result_cluster = optimized_k_cluster_centroid_and_assignment

# TODO
# make algorithm that find accurate cluster and evaluate db value so that find appropriate k
# silhouette(_centroid, _assignment, _points, _i):
k = 1
result_centroid, result_assignment = cluster_random_points(k, orbit_arr)
result_evaluated = evaluate_cluster(_centroid=result_centroid, _assignment=result_assignment, _data=orbit_arr)

while True:
    print("k is " + str(k))
    optimized_centroid, optimized_assignment = cluster_random_points(k, orbit_arr)
    optimized_evaluated = evaluate_cluster(_centroid=optimized_centroid,
                                         _assignment=optimized_assignment,
                                         _data=orbit_arr)
    curr_evaluated = optimized_evaluated
    for i in range(9):
        tmp_centroid, tmp_assignment = cluster_random_points(k, orbit_arr)
        curr_evaluated = evaluate_cluster(_centroid=tmp_centroid,
                                          _assignment=tmp_assignment,
                                          _data=orbit_arr)
        if (optimized_evaluated < curr_evaluated):
            print("changed. cluster_evaluated is " + str(optimized_evaluated) + " and curr_evaluated is " + str(curr_evaluated))
            optimized_centroid = tmp_centroid
            optimized_assignment = tmp_assignment
            optimized_evaluated = curr_evaluated
    k += 1
    cmp_cluster_evaluated = evaluate_cluster(_centroid=optimized_centroid,
                                             _assignment=optimized_assignment,
                                             _data=orbit_arr)
    if result_evaluated > optimized_evaluated :
        break
    else:
        print(k)
        result_centroid, result_assignment = optimized_centroid, optimized_assignment
        result_evaluated = optimized_evaluated
        print(result_evaluated)
k -= 2
print("result k is " + str(k))
data = {"x": [], "y": [], "cluster": []}
print(result_assignment)
for i in range(len(result_assignment)):
    data["x"].append(orbit_arr[i][0])
    data["y"].append(orbit_arr[i][1])
    data["cluster"].append(result_assignment[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()