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

def plot_data(x, y):
	colors = ['green' if val else 'red' for val in y]
	plt.scatter(x[:, 0], x[:, 1], c=colors)

def plot_linear_decision_boundary(svclassifier, x, y):
	plot_data(x, y)

	w = svclassifier.coef_[0]
	a = -w[0] / w[1]
	axx = np.linspace(-1, 5)
	axy = a * axx - (svclassifier.intercept_[0]) / w[1]
	plt.plot(axx, axy, 'k-')

	plt.show()

def gaussian_kernel(x1, x2, sigma):
	distance_length = np.sum(np.power(x1 - x2, 2))
	return np.exp(-distance_length / (2*sigma**2))

if __name__ == "__main__":

	# ------------        read data         ---------
	data = loadmat("ex5data1.mat")
	x = data['X']
	y = data['y']
	# ------------        read data         ---------



	# ------------        plot data         ---------
	# plot_data(x, y)
	# plt.show()
	# ------------        plot data         ---------



	# ------------        svm linear kernel         ---------
	# svclassifier = SVC(kernel='linear')
	# svclassifier.fit(x, y.ravel())
	# plot_linear_decision_boundary(svclassifier, x, y)
	# ------------        svm linear kernel         ---------



	# ------------        svm linear kernel C = 1        ---------
	# svclassifier = SVC(C = 1, kernel='linear')
	# svclassifier.fit(x, y.ravel())
	# plot_linear_decision_boundary(svclassifier, x, y)
	# ------------        svm linear kernel C = 1        ---------

	# ------------        svm linear kernel C = 100        ---------
	# svclassifier = SVC(C = 100, kernel='linear')
	# svclassifier.fit(x, y.ravel())
	# plot_linear_decision_boundary(svclassifier, x, y)
	# ------------        svm linear kernel C = 100        ---------




	# ------------        implement gaussian kernel        ---------
	x1 = np.array([[1, 2, 1]])
	x2 = np.array([[0, 4, -1]])
	sigma = 2
	print(gaussian_kernel(x1, x2, sigma))
	# ------------        implement gaussian kernel        ---------
