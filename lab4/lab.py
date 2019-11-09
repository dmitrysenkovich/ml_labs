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

alpha = 0.0001
l = 0
weight_sizes = [[25, 401], [10, 26]]

def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def hypothesis(x, wt):
	a1 = np.column_stack([[1]*(x.shape[0]), x])

	z2 = a1.dot(wt[0])
	a2 = sigmoid(z2)
	a2 = np.column_stack([[1]*(a2.shape[0]), a2])

	h = sigmoid(a2.dot(wt[1]))

	return h

def to_one_hot(y, size):
	converted = []
	for i in range(y.shape[0]):
		converted_i = [0] * size
		yi = y[i, 0]
		converted_i[yi - 1] = 1
		converted.append(converted_i)
	return np.array(converted)

def regularization_part(wt, m, l):
	weights_sum = regularization_weights_sum(wt[0]) + regularization_weights_sum(wt[1])
	return weights_sum * l / (2 * m)

def regularization_weights_sum(wt):
	return np.sum(np.power(wt, 2))

def cost(unrolled_weights, x, y, l):
	w = from_unrolled(unrolled_weights, weight_sizes)
	wt = [w[0].transpose(), w[1].transpose()]

	calculated_hypothesis = hypothesis(x, wt)
	m = x.shape[0]

	cost_first_part = np.multiply(y, np.log(calculated_hypothesis))
	cost_second_part = np.multiply(1 - y, np.log(1-calculated_hypothesis))

	return np.sum(cost_first_part + cost_second_part) / -m + regularization_part(wt, m, l)

def activation_derivative(al):
	return np.multiply(al, (1 - al))

def build_randomized_weights(size):
	eps = 1
	return np.random.rand(size[0], size[1]) * (2*eps) - eps

def back_propagation(unrolled_weights, x, y, l):
	w = from_unrolled(unrolled_weights, weight_sizes)
	delta1 = np.zeros(w[0].shape)
	delta2 = np.zeros(w[1].shape)
	w1 = w[0]
	w2 = w[1]
	wt1 = w1.transpose()
	wt2 = w2.transpose()
	x = np.column_stack([[1]*(x.shape[0]), x])

	for i in range(x.shape[0]):
		a1 = x[i].reshape(-1, 1).T

		z2 = a1.dot(wt1)
		a2 = sigmoid(z2)
		a2 = np.insert(a2, 0, [1]) # prepend 1

		a3 = sigmoid(a2.dot(wt2))
		yi = y[i]

		d3 = a3 - yi
		d2 = np.multiply(wt2.dot(d3), activation_derivative(a2))

		delta2 += d3.reshape(-1, 1).dot(a2.reshape(-1, 1).T)
		delta1 += d2.reshape(-1, 1)[1:, :].dot(a1)

	m = x.shape[0]
	delta1 /= m
	delta2 /= m
	delta1[:, 1:] = delta1[:, 1:] + l*w1[:, 1:]/m
	delta2[:, 1:] = delta2[:, 1:] + l*w2[:, 1:]/m

	return unroll([delta1, delta2])

def unroll(matricies):
	return np.append(matricies[0].ravel(), matricies[1].ravel())

def from_unrolled(unrolled, sizes):
	first_matrix = np.reshape(unrolled[:sizes[0][0]*sizes[0][1]], sizes[0])
	second_matrix = np.reshape(unrolled[sizes[0][0]*sizes[0][1]:], sizes[1])
	return [first_matrix, second_matrix]

def gradient_checking(w, x, y, unrolled_delta, l):
	unrolled_w = np.append(w[0].ravel(), w[1].ravel())
	n = len(unrolled_w)
	eps = 10**(-4)
	for _ in range(5):
		theta_index = int(np.random.rand() * n)
		theta_value = unrolled_w[theta_index]

		unrolled_w[theta_index] = theta_value + eps
		cost_above_derivative = cost(unrolled_w, x, y, l)

		unrolled_w[theta_index] = theta_value - eps
		cost_under_derivative = cost(unrolled_w, x, y, l)

		approximated_gradient = (cost_above_derivative - cost_under_derivative) / float(2 * eps)
		print('approximated gradient: %s, back propagation gradient: %s' % (approximated_gradient, unrolled_delta[theta_index]))

# ------------        read data         ---------
data = loadmat("ex4data1.mat")
x = data['X']
y = data['y']

weights_data = loadmat("ex4weights.mat")
w0 = weights_data['Theta1']
w1 = weights_data['Theta2']
wt = [w0.transpose(), w1.transpose()]
# number of hidden layers: 2
# size of the first layer (input layer): 400 + 1
# size of the second layer (1st hidden): 25 + 1
# size of the third layer (output layer): 10
# ------------        read data         ---------




# ------------        predict         ---------
# wt = [w0.transpose(), w1.transpose()]
# output = hypothesis(x, wt)
# predictions = [np.argmax(probabilities) + 1 for probabilities in output]

# success_count = 0
# m = x.shape[0]
# for i in range(m):
# 	prediction_i = predictions[i]
# 	yi = y[i, 0]
# 	if yi == prediction_i:
# 		success_count += 1
# print("success rate: %s" % ((success_count / m)*100)) # 97.52 vs 95.28 in logistic regression
# ------------        predict         ---------



# ------------        one hot         ---------
# yoh = to_one_hot(y, 10)
# initial_cost = cost(unroll([w0, w1]), x, yoh, 0)
# print(initial_cost)
# rand_w = build_randomized_weights([3, 2])
# print(rand_w)
# ------------        one hot         ---------





# ------------        back propagation test         ---------
# yoh = to_one_hot(y, 10)
# rand_w = [build_randomized_weights(w0.shape), build_randomized_weights(w1.shape)]
# unrolled_delta = back_propagation(unroll(rand_w), x, yoh, l)
# gradient_checking(rand_w, x, yoh, unrolled_delta, 0)
# ------------        back propagation test         ---------





# ------------        train and check network          ---------
# yoh = to_one_hot(y, 10)
# rand_w = [build_randomized_weights(w0.shape), build_randomized_weights(w1.shape)]
# unrolled_w = unroll(rand_w)
# unrolled_optimal_w = optimize.fmin_cg(maxiter=50, f=cost, x0=unrolled_w, fprime=back_propagation, args=(x, yoh, 1), disp=True)
# optimal_w = from_unrolled(unrolled_optimal_w, weight_sizes)
#
# optimal_wt = [optimal_w[0].transpose(), optimal_w[1].transpose()]
# output = hypothesis(x, optimal_wt)
# predictions = [np.argmax(probabilities) + 1 for probabilities in output]
#
# success_count = 0
# m = x.shape[0]
# for i in range(m):
# 	prediction_i = predictions[i]
# 	yi = y[i, 0]
# 	if yi == prediction_i:
# 		success_count += 1
# print("success rate: %s" % ((success_count / m)*100)) # 95.67999999999999 with 50 iterations and regularization 1
# ------------        train and check network          ---------




# ------------        best lambda        ---------
# best_lambda 0.03, min_error 0.2242020781798844
# best_lambda 0.14, max_success_rate 97.98
m = x.shape[0]
lambdas = np.linspace(0.01, 0.5, 50)
errors = []
success_rates = []
yoh = to_one_hot(y, 10)
for l in lambdas:
	rand_w = [build_randomized_weights(w0.shape), build_randomized_weights(w1.shape)]
	unrolled_w = unroll(rand_w)
	unrolled_optimal_w, error, func_calls, grad_calls, warnflag = optimize.fmin_cg(maxiter=50, f=cost, x0=unrolled_w, fprime=back_propagation, args=(x, yoh, l), disp=True, full_output=true)
	optimal_w = from_unrolled(unrolled_optimal_w, weight_sizes)
	errors.append(error)

	optimal_wt = [optimal_w[0].transpose(), optimal_w[1].transpose()]
	output = hypothesis(x, optimal_wt)
	predictions = [np.argmax(probabilities) + 1 for probabilities in output]

	success_count = 0
	m = x.shape[0]
	for i in range(m):
		prediction_i = predictions[i]
		yi = y[i, 0]
		if yi == prediction_i:
			success_count += 1

	success_rates.append(((success_count / m)*100))

best_lambda_index = np.argmin(errors)
best_lambda = lambdas[best_lambda_index]
min_error = errors[best_lambda_index]
print("best_lambda %s, min_error %s" % (best_lambda, min_error))
plt.plot(lambdas, errors, c="r")
plt.show()
best_lambda_index_by_success_rate = np.argmax(success_rates)
best_lambda = lambdas[best_lambda_index_by_success_rate]
max_success_rate = success_rates[best_lambda_index_by_success_rate]
print("best_lambda %s, max_success_rate %s" % (best_lambda, max_success_rate))
plt.plot(lambdas, success_rates, c="r")
plt.show()
#------------        best lambda        ---------





# ------------        polynomial approximation scipy        ---------
# build_polynomial_approximation(data, y, 100)
# ------------        polynomial approximation scipy       ---------


# ------------        polynomial learning curves        ---------
# build_polynomial_learning_curves(data, y, val_y, 100)
# ------------        polynomial learning curves        ---------


# ------------        best lambda approximation        ---------
# l = 0.034
# build_polynomial_approximation_for_test_set(data, y, l)
# ------------        best lambda approximation        ---------