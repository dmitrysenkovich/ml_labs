import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer
from datetime import timedelta
import scipy.optimize as optimize
from sympy import *

alpha = 0.01
max_error = 0.356
o_to_pow = []
hypothesis_x = []
hypothesis_x_transposed = []
y = []
yt = []

def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def h(ot):
	return sigmoid(hypothesis_x.dot(ot))

def j(m, ot):
	h_val = h(ot)
	return (-yt.dot(np.log(h_val)) - (1 - yt).dot(np.log(1 - h_val)))[0].item(0) / m

def j_scipy(o, x, y):
	h_val = h(o.reshape(-1, 1))
	m = hypothesis_x.shape[0]
	return ((-yt.dot(np.log(h_val)) - (1 - yt).dot(np.log(1 - h_val))))[0].item(0) / m

def j_o_der(m, ot):
	h_vec_val = h(ot)
	return (hypothesis_x_transposed.dot(h_vec_val - yt.reshape(-1, 1))) / m

def j_o_der_scipy(o, x, y):
	h_vec_val = h(o.reshape(-1, 1))
	m = hypothesis_x.shape[0]
	return (hypothesis_x_transposed.dot(h_vec_val - y.reshape(-1, 1)) / m).flatten()

def update_o(m, ot):
	j_oi_der_val = j_o_der(m, ot)
	return ot - j_oi_der_val.dot(alpha)

def gd():
	o = np.matrix([0.0] * 28)
	ot = o.transpose()
	error = 1000000000000000000
	its = 0
	m = len(hypothesis_x)
	print('m %s' % m)
	its_hist = []
	err_hist = []
	while error > max_error:
		ot = update_o(m, ot)
		error = j(m, ot)
		its += 1
		if (its % 100 == 0):
			its_hist.append(its)
			err_hist.append(error)
		if (its % 10000 == 0):
			print(error)

	print('its %s' % its)
	print('error %s' % error)
	return ot, its, its_hist, err_hist

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
		mean = sum_dimi / m
		mean_res[dimi - 1] = mean
		for i in range(m):
			x[i].itemset(dimi, ((x[i].item(dimi) - mean) / (max_dimi - min_dimi)))
		diff_res[dimi - 1] = max_dimi - min_dimi
	return mean_res, diff_res

def optimize_scipy(algorithm):
	o = np.zeros((28, 1))
	Result = optimize.minimize(fun=j_scipy, x0=o, args=(hypothesis_x, y), method=algorithm, jac=j_o_der_scipy, options={'maxiter': 6000000})
	o = Result.x
	print("o %s " % o)

def calc_o_to_pow():
	for i in range(7):
		for j in range(7 - i):
			o_to_pow.append([i, j])

def calc_hypothesis_x(x):
	m = x.shape[0]
	for i in range(m):
		xi = x[i]
		hypothesis_x.append(calc_hypothesis_x_row(xi))

def calc_hypothesis_x_row(xi):
	row = []
	for pow in o_to_pow:
		i = pow[0]
		j = pow[1]
		row.append((xi[1]**i)*(xi[2]**j))
	return row

# ------------        read data         ---------
x = []
y = []
with open('ex2data2.txt', 'r') as data_file:
	for line in data_file:
		data_i = [float(x) for x in line.split(',')]
		x.append([1, data_i[0], data_i[1]])
		y.append(data_i[2])
# ------------        read data         ---------




# ------------        plot data         ---------
# x = np.matrix(x)
# colors = ['green' if val else 'red' for val in y]
# plt.scatter(x[:,1].A1, x[:,2].A1, c=colors)
# plt.show()
# ------------        plot data         ---------



# ------------        precalculating matricies for gradient and hypothesis calculation         ---------
x = np.array(x)
y = np.array(y)
yt = y.reshape(-1, 1).transpose()
normalize_x(x)
calc_o_to_pow()
calc_hypothesis_x(x)
hypothesis_x = np.array(hypothesis_x)
hypothesis_x_transposed = hypothesis_x.transpose()
# ------------        precalculating matricies for gradient and hypothesis calculation         ---------



# ------------        gm solution      ---------
# ------------        time elapsed: 0:00:34.703448s ------
# start = timer()
# o, its, its_hist, err_hist = gd()
# end = timer()
# print(timedelta(seconds=end-start))
# print('its %s' % its)
# print('o %s' % o)
# ------------        gm solution         ---------


# ------------        scipy   BFGS      ---------
optimize_scipy('BFGS')
# ------------        scipy    BFGS     ---------


# ------------        scipy   Nelder-Mead      ---------
# optimize_scipy('Nelder-Mead')
# ------------        scipy    Nelder-Mead     ---------