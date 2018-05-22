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

# df = pd.DataFrame({"x": [v[0] for v in orbit_arr], "y": [v[1] for v in orbit_arr]})
#
# sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
# # plt.show()


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

    # DB = (1 / num_of_points)
# detect k in original algorithm

k = 2
vectors = tf.constant(orbit_arr)
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

previous_DB = David_Bouldin_index(centroid_values, assignment_values, orbit_arr)

while True:
    print(centroid_values)
    print(assignment_values)
    k += 1
    vectors = tf.constant(orbit_arr)
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

    DB = David_Bouldin_index(centroid_values, assignment_values, orbit_arr)
    if (previous_DB < DB):
        break


#
k = k - 1
print(k)
vectors = tf.constant(orbit_arr)
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

for step in range(1000):
   _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(orbit_arr[i][0])
    data["y"].append(orbit_arr[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()