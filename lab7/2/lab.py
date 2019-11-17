import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from scipy.io import loadmat
from datetime import timedelta
import math
import random

def plot_eigenvectors(u):
	sample_width = int(math.sqrt(len(u[0])))
	plot_dim = int(math.sqrt(len(u)))

	fig, axises = plt.subplots(plot_dim, plot_dim, figsize=(10, 10))
	axises = axises.ravel()

	for i in range(axises.shape[0]):
		axis = axises[i]
		axis.imshow(u[i].reshape(sample_width, sample_width, order='F'), cmap='gray')
		axis.axis('off')

def pick_n_random_indices(max=5000, n=100):
	random_indices = [0]*n
	for i in range(n):
		random_indices[i] = random.randint(0, max)
	return random_indices

def plot_n_random_faces(x, indices, n=100):
	sample_width = int(math.sqrt(len(x[0])))
	plot_dim = int(math.sqrt(n))

	fig, axises = plt.subplots(plot_dim, plot_dim, figsize=(10, 10))
	axises = axises.ravel()

	for i in range(axises.shape[0]):
		axis = axises[i]
		axis.imshow(x[indices[i]].reshape(sample_width, sample_width, order='F'), cmap='gray')
		axis.axis('off')

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
	x = loadmat("ex7faces.mat")['X']
	# ------------        read data         ---------



	# ------------        plot 100 random faces         ---------
	n = 100
	indices = pick_n_random_indices(x.shape[0], n)
	plot_n_random_faces(x, indices, n)
	plt.show()
	# ------------        plot 100 random faces         ---------



	# ------------        normalization and svd         ---------
	normalized_x, means, standard_deviations = normalize_and_scale(x)
	covariance_matrix = compute_covariance_matrix(normalized_x)
	u, s, c = np.linalg.svd(normalized_x.T)
	# ------------        normalization and svd         ---------



	# ------------        first 36 eigenvectors and compressed images         ---------
	# first k eigenvectors will give max variance as s is already sorted in descending order
	K = 36
	u_reduce = u[:, :K]
	z = project_data(u_reduce, normalized_x)
	x_approximated = recover_data(z, u_reduce)
	plot_eigenvectors(u_reduce.T)
	plt.show()
	plot_n_random_faces(x_approximated, indices, n)
	plt.show()
	# ------------        first 36 eigenvectors and compressed images         ---------



	# ------------        first 100 eigenvectors and compressed images         ---------
	# first k eigenvectors will give max variance as s is already sorted in descending order
	K = 100
	u_reduce = u[:, :K]
	z = project_data(u_reduce, normalized_x)
	x_approximated = recover_data(z, u_reduce)
	plot_eigenvectors(u_reduce.T)
	plt.show()
	plot_n_random_faces(x_approximated, indices, n)
	plt.show()
	# ------------        first 100 eigenvectors and compressed images         ---------
