import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from scipy.io import loadmat
from datetime import timedelta
import math
import random
from mpl_toolkits.mplot3d import Axes3D

def plot_image_3d(x):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x[:, 0], x[:, 1], x[:, 2])

def plot_image_2d(x):
	plt.scatter(x[:, 0], x[:, 1])

def normalize_and_scale(x):
	features_count = x.shape[1]
	means = np.zeros(shape=(features_count, 1))
	standard_deviations = np.zeros(shape=(features_count, 1))
	for i in range(features_count):
		means[i] = np.mean(x[:, i])
		standard_deviations[i] = np.std(x[:, i])

	normalized_x = (x - means.T) / standard_deviations.T

	return normalized_x, means, standard_deviations

def compute_covariance_matrix(x):
	m = x.shape[0]
	features_count = x.shape[1]
	sum = np.zeros(shape=(features_count, features_count))
	for xi in x:
		col = xi.T.reshape(features_count, 1)
		sum += col.dot(col.T)

	return sum / m

def project_data(u_reduce, x):
	return x.dot(u_reduce)

def recover_data(z, u_reduce):
	return z.dot(u_reduce.T)

if __name__ == "__main__":



	# ------------        read data         ---------
	x = loadmat('compressed.mat')['X'].reshape(-1, 3)
	# ------------        read data         ---------



	# ------------        normalization and svd         ---------
	normalized_x, means, standard_deviations = normalize_and_scale(x)
	covariance_matrix = compute_covariance_matrix(normalized_x)
	u, s, c = np.linalg.svd(normalized_x.T)
	# ------------        normalization and svd         ---------



	# ------------        plot image in 3d        ---------
	K = 3
	u_reduce = u[:, :K]
	z = project_data(u_reduce, normalized_x)
	x_approximated = recover_data(z, u_reduce)
	plot_image_3d(x_approximated)
	plt.show()
	# ------------        plot image in 3d        ---------



	# ------------        plot image in 2d        ---------
	K = 2
	u_reduce = u[:, :K]
	z = project_data(u_reduce, normalized_x)
	x_approximated = recover_data(z, u_reduce)
	plot_image_2d(x_approximated)
	plt.show()
	# ------------        plot image in 2d        ---------
