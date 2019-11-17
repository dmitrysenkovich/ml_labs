import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from scipy.io import loadmat
from datetime import timedelta

def plot_data(x, c='b'):
	plt.scatter(x[:, 0], x[:, 1], c=c)

def plot_eigenvectors(u, x):
	min_x = np.min(x[:, 0]) - 0.5
	max_x = np.max(x[:, 0]) + 0.5

	k1 = u[0][1] / u[0][0]
	ax1 = np.linspace(min_x, max_x, 100)
	ay1 = k1*ax1
	plt.plot(ax1, ay1, c='m')

	k2 = u[1][1] / u[1][0]
	ax2 = np.linspace(min_x, max_x, 100)
	ay2 = k2*ax2
	plt.plot(ax2, ay2, c='m')

def plot_projections(normalized_x, x_approximated):
	for i in range(len(normalized_x)):
		xi = normalized_x[i]
		xai = x_approximated[i]
		plt.plot([xi[0], xai[0]], [xi[1], xai[1]], c='y')

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

	K = 1



	# ------------        read data         ---------
	x = loadmat("ex7data1.mat")['X']
	# ------------        read data         ---------



	# ------------        plot data         ---------
	# plot_data(x)
	# plt.show()
	# ------------        plot data         ---------



	# ------------        normalize and scale         ---------
	# x = np.array([
	# 	[1, 2, 3],
	# 	[4, 5, 6]
	# ])
	normalized_x, means, standard_deviations = normalize_and_scale(x)
	# ------------        normalize and scale         ---------



	# ------------        build covariance matrix         ---------
	# normalized_x = np.array([
	# 	[1, 2, 3],
	# 	[4, 5, 6]
	# ])
	covariance_matrix = compute_covariance_matrix(normalized_x)
	# ------------        build covariance matrix         ---------



	# ------------        svd. u and data         ---------
	u, s, c = np.linalg.svd(normalized_x.T)
	# plot_data(normalized_x)
	# plot_eigenvectors(u, normalized_x)
	# plt.show()
	# ------------        svd. u and data         ---------



	# ------------        normalized and recovered data         ---------
	u_reduce = u[:, :K]
	z = project_data(u_reduce, normalized_x)
	x_approximated = recover_data(z, u_reduce)
	plot_data(normalized_x)
	plot_data(x_approximated, c='r')
	plot_eigenvectors(u, normalized_x)
	plot_projections(normalized_x, x_approximated)
	plt.show()
	# ------------        normalized and recovered data         ---------
