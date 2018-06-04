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

def Silhouette(_centroid, _assignment, _data):
    num_of_clusters = len(_centroid)
    s_i_list = []
    for c in range(num_of_clusters):
        a_i_list_sum = []
        b_i_list_sum = []
        a_is = []
        b_is = []
        count = (_assignment == c).sum()
        for i in range(len(_data["x"]) - 1):
            if _data["cluster"][i] == c:
                a_i_list = [distance((_data["x"][i], _data["y"][i]),(_data["x"][k], _data["y"][k])) for k in range(len(_data["x"]) - 1) if ((_data["cluster"][k] == c) and (k != i))]
                a_i_list_sum.append(sum(a_i_list) / (count - 1))
                for c2 in range(num_of_clusters):
                    if c != c2:
                        b_i_in_c2 = [distance((_data["x"][i], _data["y"][i]),(_data["x"][k], _data["y"][k])) for k in range(len(_data["x"]) - 1) if ((_data["cluster"][k] == c2))]
                        c2counter = 0
                        for t in range(len(_data["x"]) - 1):
                            if _data["cluster"][t] == c2:
                                c2counter += 1
                            
                        b_i_list_sum.append(sum(b_i_in_c2)/(c2counter))
                b_i = min(b_i_list_sum)
                a_i = sum(a_i_list_sum)/ count
                s_i = (b_i - a_i)/(max([a_i, b_i]))
                s_i_list.append(s_i)        
        
    final_s_i = sum(s_i_list) / len(_data["x"])
    print("when k = " + str(num_of_clusters) + ", s(i) = " + str(final_s_i))
    if final_s_i > 1:
        return 2
    return final_s_i
        

vectors = tf.constant(orbit_arr)
k = 2

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

S = Silhouette(centroid_values, assignment_values, data)

while True:
    k += 1

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

    previous_S = S
    S = Silhouette(centroid_values, assignment_values, data)

    if S > 1:
        k -= 1
        S = previous_S
        continue
    
    if S < previous_S :
        break

k -= 1
    
print("The value of k : " + str(k))

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
