import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
from sympy import *
from scipy.io import loadmat

alpha = 0.001
l = 0

def hypothesis(x, wt):
	return x.dot(wt)

def cost(wt, x, y):
	calculated_hypothesis = hypothesis(x, wt)
	m = x.shape[0]
	hy_diff = calculated_hypothesis - y.transpose()
	return (hy_diff.transpose().dot(hy_diff))[0].item(0) / (2 * m) + np.sum(np.power(wt[1:], 2))*l/(2*m)

def cost_scipy(w, x, y, l = 0):
	calculated_hypothesis = hypothesis(x, w.reshape(-1, 1))
	m = y.shape[1]
	hy_diff = calculated_hypothesis - y.transpose()
	return (hy_diff.transpose().dot(hy_diff))[0].item(0) / (2 * m) + np.sum(np.power(w[1:], 2))*l/(2*m)

def gradient(wt, x, y):
	calculated_hypothesis = hypothesis(x, wt)
	wt_restricted = np.copy(wt)
	wt_restricted[0][0] = 0
	m = x.shape[0]
	xt = x.transpose()
	return ((xt.dot(calculated_hypothesis - y.transpose())) / m) + wt_restricted*l/m

def gradient_scipy(w, x, y, l = 0):
	calculated_hypothesis = hypothesis(x, w.reshape(-1, 1))
	m = y.shape[1]
	w_restricted = np.copy(w).reshape(-1, 1)
	w_restricted[0] = 0
	xt = x.transpose()
	return ((xt.dot(calculated_hypothesis - y.reshape(-1, 1)) / m) + w_restricted*l/m).flatten()

def calculate_new_weights(wt, x, y):
	calculated_gradient = gradient(wt, x, y)
	return wt - calculated_gradient.dot(alpha)

def gradient_descent(x, y):
	wt = np.matrix([0.0] * x.shape[1]).transpose()
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
		if its % 100000 == 0:
			print('error %s' % error)
			return wt, its, its_hist, err_hist

def normalize_x(x):
	means = x.mean(axis=0)
	stds = np.std(x, axis=0, ddof=1)
	normalized_x = (x - means) / stds
	normalized_x = np.hstack((np.ones((len(normalized_x), 1)), normalized_x))
	return normalized_x, means, stds

def normalize_x_with_params(x, means, stds):
	normalized_x = (x - means) / stds
	normalized_x = np.hstack((np.ones((len(normalized_x), 1)), normalized_x))
	return normalized_x

def predict(x, w):
	return x.dot(w)

def build_polynomyial_x(initial_x, degree):
	polynomial_x = np.zeros(shape=(len(initial_x), degree))
	for i in range(0, degree):
		polynomial_x[:, i] = initial_x.squeeze() ** (i + 1)
	return polynomial_x

def hypothesis_polynomial_features(data, y, val_y, l):
	pol_x = build_polynomyial_x(np.array(data['X']), 8)
	pol_x, mean_res, diff_res = normalize_x(pol_x)
	w = optimize.fmin_cg(
		f=cost_scipy,
		x0=np.zeros(pol_x.shape[1]),
		fprime=gradient_scipy,
		args=(pol_x, y, l),
		maxiter=1000
	).reshape(-1, 1)

	pol_val_x = build_polynomyial_x(np.array(data['Xval']), 8)
	# normalize according to X
	pol_val_x = normalize_x_with_params(pol_val_x, mean_res, diff_res)

	return cost_scipy(w, pol_val_x, val_y)

def build_polynomial_approximation(data, y, l):
	pol_x = build_polynomyial_x(np.array(data['X']), 8)
	pol_x, mean_res, diff_res = normalize_x(pol_x)
	w = optimize.fmin_cg(
		f=cost_scipy,
		x0=np.zeros(pol_x.shape[1]),
		fprime=gradient_scipy,
		args=(pol_x, y, l),
		maxiter=1000
	).reshape(-1, 1)

	plt.scatter(pol_x[:, 1:2], np.array(data['y']))
	x_ax = np.linspace(min(np.array(data['X'])) - 5, max(np.array(data['X'])) + 5, 1000)
	# normalize according to X
	x_ax_polynomial = build_polynomyial_x(x_ax, 8)
	x_ax_polynomial = normalize_x_with_params(x_ax_polynomial, mean_res, diff_res)
	predictions = predict(x_ax_polynomial, w)
	x_ax = (x_ax - mean_res[0]) / diff_res[0]
	plt.plot(x_ax, predictions, linewidth=2)
	plt.show()

def build_polynomial_approximation_for_test_set(data, y, l):
	pol_x = build_polynomyial_x(np.array(data['X']), 8)
	pol_x, mean_res, diff_res = normalize_x(pol_x)
	w = optimize.fmin_cg(
		f=cost_scipy,
		x0=np.zeros(pol_x.shape[1]),
		fprime=gradient_scipy,
		args=(pol_x, y, l),
		maxiter=1000
	).reshape(-1, 1)

	# test set plot
	test_x_ax = (np.array(data['Xtest']) - mean_res[0]) / diff_res[0]
	plt.scatter(test_x_ax, np.array(data['ytest']))

	# predictions plot
	x_ax = np.linspace(min(np.array(data['Xtest'])) - 5, max(np.array(data['Xtest'])) + 5, 1000)
	x_ax_polynomial = build_polynomyial_x(x_ax, 8)
	x_ax_polynomial = normalize_x_with_params(x_ax_polynomial, mean_res, diff_res)
	predictions = predict(x_ax_polynomial, w)
	x_ax = (x_ax - mean_res[0]) / diff_res[0]
	plt.plot(x_ax, predictions, linewidth=2)
	plt.show()

	x_test_polynomial = build_polynomyial_x(np.array(data['Xtest']), 8)
	x_test_polynomial = normalize_x_with_params(x_test_polynomial, mean_res, diff_res)
	print(cost_scipy(w, x_test_polynomial, np.array(data['ytest'].T)))

def build_polynomial_learning_curves(data, y, val_y, l):
	m = x.shape[0]
	pol_x = build_polynomyial_x(np.array(data['X']), 8)
	pol_x, means, stds = normalize_x(pol_x)
	pol_val_x = build_polynomyial_x(np.array(data['Xval']), 8)
	pol_val_x = normalize_x_with_params(pol_val_x, means, stds)
	x_errors = []
	val_errors = []
	for i in range(2, m + 1):
		pol_xi = pol_x[0:i]
		yi = y[:, 0:i]
		w0 = np.array([0.0] * pol_x.shape[1])
		w = optimize.fmin_cg(
			f=cost_scipy,
			x0=w0,
			fprime=gradient_scipy,
			args=(pol_xi, yi, l)
		)
		x_errors.append(cost_scipy(w, pol_xi, yi))
		val_errors.append(cost_scipy(w, pol_val_x, val_y))
	x_ax = [i for i in range(2, m + 1)]
	plt.plot(x_ax, x_errors, c="g")
	plt.plot(x_ax, val_errors, c="r")
	plt.legend(["Error on training set", "Error on validation set"], loc="best")
	plt.show()

# ------------        read data         ---------
data = loadmat("ex3data1.mat")
x = np.array(data['X'])
x = np.column_stack([[1]*(x.shape[0]), x])
y = np.array(data['y']).reshape(-1, 1)
y = y.transpose()

val_x = np.array(data['Xval'])
val_x = np.column_stack([[1]*(val_x.shape[0]), val_x])
val_y = np.array(data['yval']).reshape(-1, 1)
val_y = val_y.transpose()

test_x = np.array(data['Xtest'])
test_x = np.column_stack([[1]*(test_x.shape[0]), test_x])
test_y = np.array(data['ytest']).reshape(-1, 1)
test_y = test_y.transpose()
# ------------        read data         ---------




# ------------        plot data         ---------
# plt.scatter(np.array(data['X']), np.array(data['y']))
# plt.show()
# ------------        plot data         ---------




# ------------        gm solution      ---------
# np.set_printoptions(suppress=True)
# w, its, its_hist, err_hist = gradient_descent(x, y)
# print('w %s' % w)
# plt.scatter(np.array(data['X']), np.array(data['y']))
# x_ax = np.linspace(min(np.array(data['X'])) - 5, max(np.array(data['y'])) + 5, 1000)
# x_ax = np.column_stack([[1]*(x_ax.shape[0]), x_ax])
# y_ax = predict(x_ax, w)
# plt.plot(x_ax[:, 1], y_ax, linewidth=2)
# plt.show()
# ------------        gm solution         ---------


# ------------        with scipy      ---------
# w0 = np.array([0.0]*x.shape[1])
# w = optimize.fmin_cg(
# 	f=cost_scipy,
# 	x0=w0,
# 	fprime=gradient_scipy,
# 	args=(x, y, l),
# 	maxiter=50
# )
# print('w %s' % w)
# plt.scatter(np.array(data['X']), np.array(data['y']))
# x_ax = np.linspace(min(np.array(data['X'])) - 5, max(np.array(data['y'])) + 5, 1000)
# x_ax = np.column_stack([[1]*(x_ax.shape[0]), x_ax])
# y_ax = predict(x_ax, w)
# plt.plot(x_ax[:, 1], y_ax, linewidth=2)
# plt.show()
# ------------        with scipy      ---------


# ------------        learning curves line        ---------
# l = 0
# m = x.shape[0]
# x_errors = []
# val_errors = []
# for i in range(2, m+1):
# 	xi = x[0:i]
# 	yi = y[:, 0:i]
# 	w0 = np.array([0.0] * x.shape[1])
# 	w = optimize.fmin_cg(
# 		f=cost_scipy,
# 		x0=w0,
# 		fprime=gradient_scipy,
# 		args=(xi, yi, l)
# 	)
# 	x_errors.append(cost_scipy(w, xi, yi))
# 	val_errors.append(cost_scipy(w, val_x, val_y))
# x_ax = [i for i in range(2, m+1)]
# plt.plot(x_ax, x_errors, c="g")
# plt.plot(x_ax, val_errors, c="r")
# plt.legend(["Error on training set", "Error on validation set"], loc="best")
# plt.show()
# ------------        learning curves line        ---------





# ------------        polynomial approximation        ---------
# l = 0
# pol_x = build_polynomyial_x(np.array(data['X']), 8)
# pol_x, mean_res, diff_res = normalize_x(pol_x)
# w, its, its_hist, err_hist = gradient_descent(pol_x, y)
# plt.scatter(pol_x[:, 1:2], np.array(data['y']))
# x_ax = np.linspace(min(np.array(data['X'])) - 5, max(np.array(data['X'])) + 5, 1000)
# # normalize according to X
# x_ax_polynomial = build_polynomyial_x(x_ax, 8)
# x_ax_polynomial = normalize_x_with_params(x_ax_polynomial, mean_res, diff_res)
# predictions = predict(x_ax_polynomial, w)
# x_ax = (x_ax - mean_res[0]) / diff_res[0]
# plt.plot(x_ax, predictions, linewidth=2)
# plt.show()
# plt.plot(its_hist, err_hist)
# plt.show()
# ------------        polynomial approximation       ---------





# ------------        polynomial approximation scipy        ---------
# build_polynomial_approximation(data, y, 100)
# ------------        polynomial approximation scipy       ---------


# ------------        polynomial learning curves        ---------
# build_polynomial_learning_curves(data, y, val_y, 100)
# ------------        polynomial learning curves        ---------




# ------------        best lambda        ---------
# best_lambda 2.972, min_error 3.822768364932697
# lambdas = np.linspace(0.001, 10, 10000)
# test_errors = []
# for l in lambdas:
# 	err = hypothesis_polynomial_features(data, y, val_y, l)
# 	test_errors.append(err)
# best_lambda_index = np.argmin(test_errors)
# best_lambda = lambdas[best_lambda_index]
# min_error = test_errors[best_lambda_index]
# print("best_lambda %s, min_error %s" % (best_lambda, min_error))
# plt.plot(lambdas, test_errors, c="r")
# plt.show()
#------------        best lambda        ---------


# ------------        best lambda approximation        ---------
l = 2.972
build_polynomial_approximation_for_test_set(data, y, l)
# ------------        best lambda approximation        ---------