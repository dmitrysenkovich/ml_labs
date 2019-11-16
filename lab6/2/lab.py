import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import scipy.optimize as optimize
from scipy.io import loadmat
from datetime import timedelta
import random
from imageio import imread

k_means_iterations_count = 50
algorithm_iterations_count = 10

def cost(x, centroids, centroid_indices_per_sample):
	K = centroids.shape[0]
	m = x.shape[0]
	cost = 0
	for k in range(K):
		centroid = centroids[k]
		neighbours = pick_centroid_neighbours(x, centroid_indices_per_sample, k)
		if not len(neighbours):
			continue
		distance = np.power(neighbours - centroid.reshape(1, -1), 2).sum(axis=1)
		cost += distance.sum(axis=0)
	return cost / m

def assign_random_centroids(x, K):
	m = x.shape[0]
	centroid_indices = random.sample(range(1, m), K)
	return np.array([x[centroid_index] for centroid_index in centroid_indices])

def pick_centroid_neighbours(x, centroid_indices_per_sample, centroid_index):
	centroid_neighbours_indices = [i for i in range(x.shape[0]) if centroid_indices_per_sample[i] == centroid_index]
	return np.array([x[i] for i in centroid_neighbours_indices])

def k_means(x, K):
	last_centroids = assign_random_centroids(x, K)
	m = x.shape[0]
	centroid_indices_per_sample = [0] * m

	for _ in range(k_means_iterations_count):
		for i in range(m):
			xi = x[i]
			distance = np.power(last_centroids - xi.reshape(1, -1), 2).sum(axis=1)
			closest_centroid_index = np.argmin(distance)
			centroid_indices_per_sample[i] = closest_centroid_index

		centroids = np.zeros(shape = (K, x.shape[1]))
		for k in range(K):
			neighbours = pick_centroid_neighbours(x, centroid_indices_per_sample, k)
			centroids[k] = neighbours.sum(axis=0) / (1 if len(neighbours) == 0 else len(neighbours))

		last_centroids = centroids

	return last_centroids, cost(x, last_centroids, centroid_indices_per_sample), centroid_indices_per_sample

def find_best_clusterization(x, K):
	best_cost = 100000000000000000000
	best_centroids = []
	best_centroid_indices_per_sample = []
	current_iteration_number = 1
	for _ in range(algorithm_iterations_count):
		centroids, cost, centroid_indices_per_sample = k_means(x, K)
		if cost < best_cost:
			best_cost = cost
			best_centroids = centroids
			best_centroid_indices_per_sample = centroid_indices_per_sample
		print(current_iteration_number)
		current_iteration_number += 1
	return np.rint(best_centroids).astype(int), best_cost, best_centroid_indices_per_sample

def plot_image(x):
	plt.imshow(x)
	plt.show()

def build_compressed_image(centroids, centroid_indices_per_sample):
	image_size = 128
	compressed = np.zeros(shape = (image_size*image_size, 3)).astype(int)
	for i in range(len(centroid_indices_per_sample)):
		centroid = centroids[centroid_indices_per_sample[i]]
		compressed[i] = centroid
	return compressed.reshape(image_size, image_size, 3)

if __name__ == "__main__":



	# ------------        read data         ---------
	# x = loadmat("bird_small.mat")['A']
	# ------------        read data         ---------



	# ------------        plot data         ---------
	# plot_image(x)
	# ------------        plot data         ---------



	# ------------        best clusterization         ---------
	# best_centroids, best_cost, best_centroid_indices_per_sample = find_best_clusterization(x.reshape((-1, 3)), 16)
	# print('best cost %s' % best_cost)
	# compressed = build_compressed_image(best_centroids, best_centroid_indices_per_sample)
	# plot_image(compressed)
	# best cost 467.1932244020061
	# ------------        best clusterization         ---------



	# ------------        my image to mat conversion and test         ---------
	x = imread('my_original.jpg')
	plot_image(x)
	best_centroids, best_cost, best_centroid_indices_per_sample = find_best_clusterization(x.reshape((-1, 3)), 16)
	print('best cost %s' % best_cost)
	compressed = build_compressed_image(best_centroids, best_centroid_indices_per_sample)
	plot_image(compressed)
	# ------------        my image to mat conversion and test         ---------
