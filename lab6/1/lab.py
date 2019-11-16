import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import scipy.optimize as optimize
from scipy.io import loadmat
from datetime import timedelta
import random

k_means_iterations_count = 5
algorithm_iterations_count = 100
centroid_colors = ['r', 'g', 'b']

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
	centroid_history = [[last_centroids[0]], [last_centroids[1]], [last_centroids[2]]]

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
			centroid_history[k].append(centroids[k])

		last_centroids = centroids

	for k in range(K):
		centroid_history[k] = np.array(centroid_history[k])

	return last_centroids, centroid_history, cost(x, last_centroids, centroid_indices_per_sample), centroid_indices_per_sample

def find_best_clusterization(x, K):
	best_cost = 100000000000000000000
	best_centroids = []
	best_centroid_history = []
	best_centroid_indices_per_sample = []
	for _ in range(algorithm_iterations_count):
		centroids, centroid_history, cost, centroid_indices_per_sample = k_means(x, K)
		if cost < best_cost:
			best_cost = cost
			best_centroids = centroids
			best_centroid_history = centroid_history
			best_centroid_indices_per_sample = centroid_indices_per_sample
	return best_centroids, best_centroid_history, best_cost, best_centroid_indices_per_sample

def plot_data_with_centroids(x, centroids):
	plt.scatter(x[:, 0], x[:, 1])
	plt.scatter(centroids[:, 0], centroids[:, 1], c='g')
	plt.show()

def plot_clusters_colorized(x, centroid_indices_per_sample, centroids):
	for k in range(3):
		color = centroid_colors[k]
		neighbours = pick_centroid_neighbours(x, centroid_indices_per_sample, k)
		plt.scatter(neighbours[:, 0], neighbours[:, 1], c=color)
		centroid = centroids[k]
		plt.scatter(centroid[0], centroid[1], c=color, s=150)
	plt.show()

def plot_centroid_trajectory(centroid_history, k):
	labels = [*range(1, len(centroid_history[0] + 1))]
	plt.plot(centroid_history[k][:, 0], centroid_history[k][:, 1], c=centroid_colors[k])
	for i in range(len(labels)):
		label = str(labels[i])
		plt.text(centroid_history[k][i, 0], centroid_history[k][i, 1], label, fontsize=9)
	plt.show()

def plot_centroids_trajectory(centroid_history):
	labels = [*range(1, len(centroid_history[0] + 1))]
	for k in range(3):
		plt.plot(centroid_history[k][:, 0], centroid_history[k][:, 1], c=centroid_colors[k])
		for i in range(len(labels)):
			label = str(labels[i])
			plt.text(centroid_history[k][i, 0], centroid_history[k][i, 1], label, fontsize=9)
	plt.show()

if __name__ == "__main__":



	# ------------        read data         ---------
	data = loadmat("ex6data1.mat")
	x = data['X']
	# ------------        read data         ---------



	# ------------        plot data         ---------
	# plt.scatter(x[:, 0], x[:, 1])
	# plt.show()
	# ------------        plot data         ---------




	# ------------        k means         ---------
	# x = np.array([
	# 	[1, 1], [2, 2], [5, 5], [6, 6], [9, 9], [10, 10]
	# ])
	# centroids, centroid_history, cost, centroid_indices_per_sample = k_means(x, 3)
	# print(cost)
	#
	# # data and cluster centroids
	# plot_data_with_centroids(x, centroids)
	#
	# # clusters in colors
	# plot_clusters_colorized(x, centroid_indices_per_sample, centroids)

	# plt.plot(centroid_history[0][:, 0], centroid_history[0][:, 1], c='b')
	# plt.scatter(centroid_history[0][:, 0], centroid_history[0][:, 1], c='b')
	#
	# plt.plot(centroid_history[1][:, 0], centroid_history[1][:, 1], c='y')
	# plt.scatter(centroid_history[1][:, 0], centroid_history[1][:, 1], c='y')
	#
	# plt.plot(centroid_history[2][:, 0], centroid_history[2][:, 1], c='r')
	# plt.scatter(centroid_history[2][:, 0], centroid_history[2][:, 1], c='r')
	# ------------        k means         ---------



	# ------------        best clusterization         ---------
	best_centroids, best_centroid_history, best_cost, best_centroid_indices_per_sample = find_best_clusterization(x, 3)
	print('best cost %s' % best_cost)
	plot_data_with_centroids(x, best_centroids)
	plot_clusters_colorized(x, best_centroid_indices_per_sample, best_centroids)
	plot_centroids_trajectory(best_centroid_history)
	for k in range(3):
		plot_centroid_trajectory(best_centroid_history, k)
	# ------------        best clusterization         ---------
