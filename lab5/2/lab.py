import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer
from datetime import timedelta
import scipy.optimize as optimize
from sympy import *
from scipy.io import loadmat
import pandas as pd
from sklearn.svm import SVC
from datetime import timedelta

def plot_data(x, y):
	colors = ['green' if val else 'red' for val in y]
	plt.scatter(x[:, 0], x[:, 1], c=colors)

def plot_nonlinear_decision_boundary(svclassifier, x, y, sigma):
	x_min, x_max = x[:, 0].min(), x[:, 0].max()
	y_min, y_max = x[:, 1].min(), x[:, 1].max()
	ax = np.linspace(x_min, x_max, 100)
	ay = np.linspace(y_min, y_max, 100)
	xx = np.array([(axx, ayy) for axx in ax for ayy in ay])

	xx_in_f = to_f_space(xx, x, sigma)
	xx_predictions = svclassifier.predict(xx_in_f)

	colors = ['blue' if prediction else 'yellow' for prediction in xx_predictions]
	plt.scatter(xx[:, 0], xx[:, 1], c=colors, s = 10)

	plot_data(x, y)

	plt.show()

def gaussian_kernel_single_cell(x1, x2, sigma):
	distance_length = np.sum(np.power(x1 - x2, 2))
	return np.exp(-distance_length / (2*sigma**2))

def gaussian_kernel(x, sigma):
	m = x.shape[0]
	kernel = np.zeros((m, m))
	for i in range(m):
		li = x[i]
		for j in range(m):
			xj = x[j]
			kernel[j, i] = gaussian_kernel_single_cell(xj, li, sigma)
	return kernel

def to_f_space(xx, x, sigma):
	m = x.shape[0]
	xxm = xx.shape[0]
	in_f_space = np.zeros((xxm, m))
	for i in range(m):
		li = x[i]
		for j in range(xxm):
			xj = xx[j]
			in_f_space[j, i] = gaussian_kernel_single_cell(xj, li, sigma)

	return in_f_space

if __name__ == "__main__":

	# ------------        read data         ---------
	data = loadmat("ex5data2.mat")
	x = data['X']
	y = data['y']
	# ------------        read data         ---------



	# ------------        plot data         ---------
	# plot_data(x, y)
	# plt.show()
	# ------------        plot data         ---------



	# ------------        svm gaussian kernel c = 1         ---------
	# sigma = 0.1
	# f = gaussian_kernel(x, sigma)
	# svclassifier = SVC(C = 1, kernel = 'linear')
	# svclassifier.fit(f, y.ravel())
	# plot_nonlinear_decision_boundary(svclassifier, x, y, sigma)
	# ------------        svm gaussian kernel c = 100         ---------



	# ------------        svm gaussian kernel c = 1         ---------
	start = timer()
	sigma = 0.1
	f = gaussian_kernel(x, sigma)
	svclassifier = SVC(C = 100, kernel = 'linear')
	svclassifier.fit(f, y.ravel())
	plot_nonlinear_decision_boundary(svclassifier, x, y, sigma)
	end = timer()
	print(timedelta(seconds=end-start))
	# ------------        svm gaussian kernel c = 100         ---------
