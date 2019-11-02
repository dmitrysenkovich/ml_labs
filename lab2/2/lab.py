import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer
from datetime import timedelta
import scipy.optimize as optimize
from sympy import *

# at first max error is 0.45 starting with l = 1 max errors are 0.53, 0.6482157015, 0.6864838338726167, 0.6923102938884204
alpha = 0.001
l = 1000
max_error = 0.6923102938884204
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
	return ((-yt.dot(np.log(h_val)) - (1 - yt).dot(np.log(1 - h_val)))[0].item(0) / m) + np.sum(np.power(ot[1:], 2))*l/(2*m)

def j_scipy(o, x, y):
	h_val = h(o.reshape(-1, 1))
	m = hypothesis_x.shape[0]
	return ((-yt.dot(np.log(h_val)) - (1 - yt).dot(np.log(1 - h_val)))[0].item(0) / m) + np.sum(np.power(o[1:], 2))*l/(2*m)

def j_o_der(m, ot):
	h_vec_val = h(ot)
	ot_restricted = np.copy(ot)
	ot_restricted[0][0] = 0
	return ((hypothesis_x_transposed.dot(h_vec_val - y)) / m) + ot_restricted*l/m

def j_o_der_scipy(o, x, y):
	h_vec_val = h(o.reshape(-1, 1))
	m = hypothesis_x.shape[0]
	o_restricted = np.copy(o).reshape(-1, 1)
	o_restricted[0] = 0
	return ((hypothesis_x_transposed.dot(h_vec_val - y.reshape(-1, 1)) / m) + o_restricted*l/m).flatten()

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
	print(j(m, ot))
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
	Result = optimize.minimize(fun=j_scipy, x0=o, args=(hypothesis_x, y), method=algorithm, jac=j_o_der_scipy, options={'maxiter': 10000000})
	o = Result.x
	print("o %s " % o)
	return o

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
		row.append((xi[0]**i)*(xi[1]**j))
	return row

# ------------        read data         ---------
x = []
y = []
with open('ex2data2.txt', 'r') as data_file:
	for line in data_file:
		data_i = [float(x) for x in line.split(',')]
		x.append([data_i[0], data_i[1]])
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
y = np.array(y).reshape(-1, 1)
yt = y.transpose()
#normalize_x(x)
calc_o_to_pow()
calc_hypothesis_x(x)
hypothesis_x = np.array(hypothesis_x)
hypothesis_x_transposed = hypothesis_x.transpose()
# ------------        precalculating matricies for gradient and hypothesis calculation         ---------



# ------------        gm solution      ---------
# ------------        time elapsed: 0:00:34.703448s ------
start = timer()
o, its, its_hist, err_hist = gd()
end = timer()
print(timedelta(seconds=end-start))
np.set_printoptions(suppress=True)
print('its %s' % its)
print('o %s' % o)
# ------------        gm solution         ---------


# ------------        scipy   BFGS      ---------
# o = optimize_scipy('BFGS')
# ------------        scipy    BFGS     ---------


# ------------        scipy   Nelder-Mead      ---------
# o = optimize_scipy('Nelder-Mead')
# ------------        scipy    Nelder-Mead     ---------



# ------------        plot predictions vs initial data     ---------
t = np.linspace(-1, 1.25, 100)
asd = []
for tt in t:
	for ttt in t:
		asd.append([tt, ttt])
t = np.array(asd)
hypothesis_x = []
calc_hypothesis_x(t)
hypothesis_x = np.array(hypothesis_x)
values = sigmoid(hypothesis_x.dot(o)) >= 0.5
colors_v = ['blue' if val else 'yellow' for val in values]
plt.scatter(t[:,0], t[:,1], c=colors_v)
colors = ['green' if val else 'red' for val in y]
plt.scatter(x[:,0], x[:,1], c=colors)
plt.show()
# ------------        plot predictions vs initial data     ---------