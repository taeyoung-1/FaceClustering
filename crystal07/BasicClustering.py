import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import tensorflow as tf

def get_total_dunn(data) :
    data_per_cluster = {}
    index = 0
    for i in data["cluster"] : # 0->x, 1->y, 2->cluster
        if i in data_per_cluster.keys() :
            data_per_cluster[i].append([data["x"][index], data["y"][index]])
        else :
            data_per_cluster[i] = [[data["x"][index], data["y"][index]]]
        index = index + 1

    dunn_list = []
    for i in data_per_cluster.keys() :
        for j in data_per_cluster.keys() :
            if list(data_per_cluster.keys()).index(i) >= list(data_per_cluster.keys()).index(j) :
                continue

            dunn_list.append(get_dunn(data_per_cluster[i], list(data_per_cluster.values())))

    if len(dunn_list) == 0 :
        return -1
    else :
        return sum(dunn_list) / len(dunn_list)


def get_dunn(list_x, list_y) :
    distance = []
    for i in list_y :
        if list_x == i :
            continue

        distance.append(get_distance_per_cluster(list_x, i))
    return get_distance_in_cluster(list_x) / min(distance)


def get_distance_per_cluster(cluster_x, cluster_y) :
    avg_x_x = sum(list(map(lambda avg_x : avg_x[0], cluster_x))) / len(cluster_x)
    avg_x_y = sum(list(map(lambda avg_y : avg_y[1], cluster_x))) / len(cluster_x)

    avg_y_x = sum(list(map(lambda avg_x : avg_x[0], cluster_y))) / len(cluster_y)
    avg_y_y = sum(list(map(lambda avg_y : avg_y[1], cluster_y))) / len(cluster_y)

    return (((avg_x_x-avg_y_x)**2+(avg_x_y-avg_y_y)**2)**0.5)

def get_distance_in_cluster(cluster) :
    distance_in_cluster = []

    for i in cluster :
        for j in cluster :
            if cluster.index(i) >= cluster.index(j) :
                continue

            distance_in_cluster.append((((i[1]-j[1])**2+(i[0]-j[0])**2)**0.5))

    if len(distance_in_cluster) <= 0 : #Laplacian
        return 0 + 1
    else :
        return max(distance_in_cluster) + 1 

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
max_k = 0
max_dunn = -500 #evaluate by dunn index

vectors = tf.constant(orbit_arr)
for i in range(2, 11) :
    k = i
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

    dunn = get_total_dunn(data)
    print("dunn is", dunn, "k is", k)
    if dunn > max_dunn :
        max_dunn = dunn
        max_k = k

    df = pd.DataFrame(data)
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
    plt.show()

print("best k is", max_k)