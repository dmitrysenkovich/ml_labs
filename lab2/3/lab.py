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

alpha = 0.001
l = 0
x = []
xt = []

def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def hypothesis(wt):
	return sigmoid(x.dot(wt))

def cost(wt, yt):
	calculated_hypothesis = hypothesis(wt)
	m = x.shape[0]
	return ((-yt.dot(np.log(calculated_hypothesis)) - (1 - yt).dot(np.log(1 - calculated_hypothesis)))[0].item(0) / m) + np.sum(np.power(wt[1:], 2))*l/(2*m)

def cost_scipy(w, x, y, ll):
	calculated_hypothesis = hypothesis(w.reshape(-1, 1))
	m = x.shape[0]
	yt = y.reshape(-1, 1).transpose()
	return ((-yt.dot(np.log(calculated_hypothesis)) - (1 - yt).dot(np.log(1 - calculated_hypothesis)))[0].item(0) / m) + np.sum(np.power(w[1:], 2))*l/(2*m)

def gradient(wt, y):
	calculated_hypothesis = hypothesis(wt)
	wt_restricted = np.copy(wt)
	wt_restricted[0][0] = 0
	m = x.shape[0]
	return ((xt.dot(calculated_hypothesis - y)) / m) + wt_restricted*l/m

def gradient_scipy(w, x, y, ll):
	calculated_hypothesis = hypothesis(w.reshape(-1, 1))
	m = x.shape[0]
	w_restricted = np.copy(w).reshape(-1, 1)
	w_restricted[0] = 0
	return ((xt.dot(calculated_hypothesis - y.reshape(-1, 1)) / m) + w_restricted*l/m).flatten()

def calculate_new_weights(wt, y):
	calculated_gradient = gradient(wt, y)
	return wt - calculated_gradient.dot(alpha)

def gradient_descent(y):
	yt = y.transpose()
	w = np.matrix([0.0] * 401)
	wt = w.transpose()
	its = 0
	its_hist = []
	err_hist = []
	while true:
		wt = calculate_new_weights(wt, y)
		error = cost(wt, yt)
		its += 1
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

def is_n(classifier, xi):
	return sigmoid(classifier.dot(xi.reshape(-1, 1)))[0].item(0)

# ------------        read data         ---------
data = loadmat("ex2data3.mat")
x = np.array(data['X'])
x = np.column_stack([[1]*(x.shape[0]), x])
xt = x.transpose()
target = np.array(data['y']).reshape(-1, 1)
target_t = target.transpose()
# plt.imshow(x[0].reshape(20, 20), cmap='hot')
# plt.show()
# ------------        read data         ---------




# ------------        plot data         ---------
# data = loadmat("ex2data3.mat")
# x = data['X']
# target = data['y']
# fig, axs = plt.subplots(1, 10)
# t = -1
# i = 0
# for j in range(target.shape[0]):
# 	if target[j] != t:
# 		axs[i].imshow(x[j].reshape(20, 20), cmap='hot')
# 		axs[i].axis("off")
# 		i += 1
# 		t = target[j]
# 		print(t)
# plt.show()
# ------------        plot data         ---------




# ------------        gm solution      ---------
# classifiers = []
# np.set_printoptions(suppress=True)
# index_to_number = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# for i in range(10):
# 	print('--------------')
# 	print('number is %s' % (i+1))
# 	y = [1 if val == index_to_number[i] else 0 for val in target]
# 	y = np.array(y).reshape(-1, 1)
# 	w, its, its_hist, err_hist = gradient_descent(y)
# 	classifiers.append(w.transpose())
# 	print('--------------')
# print(len(classifiers))
# ------------        gm solution         ---------


# ------------        with scipy      ---------
classifiers = []
index_to_number = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(10):
	y = [1 if val == index_to_number[i] else 0 for val in target]
	y = np.array(y).reshape(-1, 1)
	w0 = np.array([0.0]*401)
	classifiers.append(optimize.fmin_cg(
		f=cost_scipy,
		x0=w0,
		fprime=gradient_scipy,
		args=(x, y, l),
		maxiter=100
	))
# ------------        with scipy      ---------


# ------------        check classifiers         ---------
m = x.shape[0]
predictions = [0]*m
classifier_to_prediction = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(m):
	xi = x[i]
	probabilities = [0]*10
	for k in range(10):
		classifier = classifiers[k]
		probabilities[k] = is_n(classifier, xi)
	prediction = classifier_to_prediction[np.argmax(probabilities)]
	predictions[i] = prediction

success_count = 0
for i in range(m):
	prediction_i = predictions[i]
	target_i = target[i].item(0)
	success_count += 1 if target_i == prediction_i else 0
print("success rate: %s" % ((success_count / m)*100)) # 91.06 in 2m its without scipy; 95.28 in 50 its with scipy

# ------------        check classifiers         ---------
