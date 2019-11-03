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

alpha = 0.00001
l = 0

def hypothesis(x, wt):
	return x.dot(wt)

def cost(wt, x, y):
	calculated_hypothesis = hypothesis(x, wt)
	m = x.shape[0]
	hy_diff = calculated_hypothesis - y.transpose()
	return (hy_diff.transpose().dot(hy_diff))[0].item(0) / (2 * m) + np.sum(np.power(wt[1:], 2))*l/(2*m)

def cost_scipy(w, x, y, l):
	calculated_hypothesis = hypothesis(x, w.reshape(-1, 1))
	m = x.shape[0]
	hy_diff = calculated_hypothesis - y.transpose()
	return (hy_diff.transpose().dot(hy_diff))[0].item(0) / (2 * m) + np.sum(np.power(w[1:], 2))*l/(2*m)

def gradient(wt, x, y):
	calculated_hypothesis = hypothesis(x, wt)
	wt_restricted = np.copy(wt)
	wt_restricted[0][0] = 0
	m = x.shape[0]
	xt = x.transpose()
	return ((xt.dot(calculated_hypothesis - y.transpose())) / m) + wt_restricted*l/m

def gradient_scipy(w, x, y, l):
	calculated_hypothesis = hypothesis(x, w.reshape(-1, 1))
	m = x.shape[0]
	w_restricted = np.copy(w).reshape(-1, 1)
	w_restricted[0] = 0
	xt = x.transpose()
	return ((xt.dot(calculated_hypothesis - y.reshape(-1, 1)) / m) + w_restricted*l/m).flatten()

def calculate_new_weights(wt, x, y):
	calculated_gradient = gradient(wt, x, y)
	return wt - calculated_gradient.dot(alpha)

def gradient_descent(x, y):
	wt = np.matrix([0.0] * 2).transpose()
	its = 0
	its_hist = []
	err_hist = []
	while true:
		wt = calculate_new_weights(wt, x, y)
		error = cost(wt, x, y)
		its += 1
		if its % 10000 == 0:
			print('error %s' % error)
			its_hist.append(its)
			err_hist.append(error)
		if its % 200000 == 0:
			print('error %s' % error)
			return wt, its, its_hist, err_hist

def normalize_x(x):
	dim = x[0].size
	m = len(x)
	mean_res = [0] * (dim - 1)
	diff_res = [0] * (dim - 1)
	for dimi in range(1, dim):
		sum_dimi = 0.0
		min_dimi = float('inf')
		max_dimi = float('-inf')
		for i in range(m):
			sum_dimi += x[i].item(dimi)
			if x[i].item(dimi) < min_dimi:
				min_dimi = x[i].item(dimi)
			if x[i].item(dimi) > max_dimi:
				max_dimi = x[i].item(dimi)
		if (sum == 0) or (min_dimi == max_dimi):
			continue
		mean = sum_dimi / m
		mean_res[dimi - 1] = mean
		for i in range(m):
			x[i].itemset(dimi, ((x[i].item(dimi) - mean) / (max_dimi - min_dimi)))
		diff_res[dimi - 1] = max_dimi - min_dimi
	return mean_res, diff_res

def predict(x, w):
	return x*w

# ------------        read data         ---------
data = loadmat("ex3data1.mat")
x = np.array(data['Xval'])
#normalize_x(x)
x = np.column_stack([[1]*(x.shape[0]), x])
val_y = np.array(data['yval']).reshape(-1, 1)
val_y = val_y.transpose()
test_x = np.array(data['Xtest'])
test_x = np.column_stack([[1]*(test_x.shape[0]), test_x])
test_y = np.array(data['ytest']).reshape(-1, 1)
# ------------        read data         ---------




# ------------        plot data         ---------
# plt.scatter(np.array(data['Xval']), np.array(data['yval']))
# plt.show()
# ------------        plot data         ---------




# ------------        gm solution      ---------
# np.set_printoptions(suppress=True)
# w, its, its_hist, err_hist = gradient_descent(x, val_y)
# print('w %s' % w)
# ------------        gm solution         ---------


# ------------        with scipy      ---------
w0 = np.array([0.0]*2)
w = optimize.fmin_cg(
	f=cost_scipy,
	x0=w0,
	fprime=gradient_scipy,
	args=(x, val_y, l),
	maxiter=50
)
print('w %s' % w)
# ------------        with scipy      ---------


# ------------        learning curves        ---------
m = x.shape[0]
errors = []
for i in range(m):
	pass
# ------------        learning curves        ---------


# ------------        plot data with hypothesis        ---------
# plt.scatter(np.array(data['Xval']), np.array(data['yval']))
# points_x = np.linspace(-50, 50, 1000)
# points_y = w[0].item(0) + w[1].item(0)*points_x
# plt.plot(points_x, points_y, '-r')
# plt.show()
# ------------        plot data with hypothesis         ---------


# ------------        check classifiers         ---------
# m = x.shape[0]
# predictions = [0]*m
# classifier_to_prediction = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# for i in range(m):
# 	xi = x[i]
# 	probabilities = [0]*10
# 	for k in range(10):
# 		classifier = classifiers[k]
# 		probabilities[k] = is_n(classifier, xi)
# 	prediction = classifier_to_prediction[np.argmax(probabilities)]
# 	predictions[i] = prediction
#
# success_count = 0
# for i in range(m):
# 	prediction_i = predictions[i]
# 	target_i = target[i].item(0)
# 	success_count += 1 if target_i == prediction_i else 0
# print("success rate: %s" % ((success_count / m)*100)) # 91.06 in 2m its without scipy; 95.28 in 50 its with scipy

# ------------        check classifiers         ---------
