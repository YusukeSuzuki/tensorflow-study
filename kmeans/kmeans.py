import tensorflow as tf
import numpy as np
import sys

# simple k-means clustering with tensorflow
# this is my practice code for tensorflow graph coding

# parameters

K_SIZE = 5

SAMPLE_SIZE = 10000
SAMPLES_COV = [[5,0],[0,5]]
SAMPLES_MEAN_PARAM = 100

# create samples and initial centers

sample_list = []

for i in range(K_SIZE):
    random_center = (SAMPLES_MEAN_PARAM * np.random.random_sample(2) - SAMPLES_MEAN_PARAM / 2).astype(np.float32)
    sample_list.append(np.random.multivariate_normal(random_center, SAMPLES_COV, [SAMPLE_SIZE, 2]).astype(np.float32))

samples_data = np.reshape(sample_list,[-1,2])
np.random.shuffle(samples_data)
center_list = samples_data[0:5]
centers_data = np.reshape(center_list,[-1,2])

# make tensorflow graph

samples = tf.Variable(samples_data)
centers = tf.placeholder(tf.float32)

centers_map = tf.tile(centers, [samples_data.shape[0],1])
centers_map = tf.reshape(centers_map,[samples_data.shape[0] * K_SIZE, 2])

samples_map = tf.tile(samples, [1,K_SIZE])
samples_map = tf.reshape(samples_map,[samples_data.shape[0] * K_SIZE, 2])

distance_map = tf.sub(centers_map, samples_map)
distance_map = tf.pow(distance_map, [2])
distance_map = tf.reduce_sum(distance_map, 1)
distance_map = tf.reshape(distance_map, [-1, K_SIZE])

labels = tf.to_int32(tf.argmin(distance_map, 1))

# centers update iteration

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

updated_samples_list = tf.dynamic_partition(samples, labels, K_SIZE)
updated_centers = tf.concat(0,[tf.reduce_mean(t,0) for t in updated_samples_list])
updated_centers = tf.reshape(updated_centers, [-1,2])

for i in range(100):
    centers_data = sess.run(updated_centers, feed_dict={centers:centers_data})

labeled_samples = sess.run(updated_samples_list, feed_dict={centers: centers_data})

# output clustered samples with gnuplot points format

index = 0
command = 'plot '

for s in labeled_samples:
    if not len(s):
        continue

    command = command + ' "out.txt" using 1:2 index {0} with points,'.format(index)
    index = index + 1

    for p in s:
        print('{0} {1}'.format(p[0],p[1]))
    print('')
    print('')

# output plot command to stderr

print(command[0:-1], file=sys.stderr)

sess.close()

