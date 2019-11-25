import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer
from datetime import timedelta
import scipy.optimize as optimize

alpha = 0.01
max_error = 0.208

def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def h(ot, x):
	return sigmoid(x.dot(ot))

def j(x, y, m, ot):
	h_val = h(ot, x)
	return (-y * np.log(h_val) - (1 - y) * np.log(1 - h_val))[0].item(0) / m

def j_scipy(o, x, y):
	h_val = h(o.reshape(-1, 1), x)
	m = x.shape[0]
	yt = y.reshape(-1, 1).transpose()
	return ((-yt.dot(np.log(h_val)) - (1 - yt).dot(np.log(1 - h_val))))[0].item(0) / m

def j_oi_der(x, xt, y, yt, m, ot):
	h_vec_val = h(ot, x)
	return (xt * (h_vec_val - yt)) / m

def j_o_der_scipy(o, x, y):
	h_vec_val = h(o.reshape(-1, 1), x)
	m = x.shape[0]
	return (x.transpose().dot(h_vec_val - y.reshape(-1, 1)) / m).flatten()

def update_o(x, xt, y, yt, m, ot):
	j_oi_der_val = j_oi_der(x, xt, y, yt, m, ot)
	return ot - alpha * j_oi_der_val

def gd(x, y):
	o = np.matrix([0.0, 0.0, 0.0])
	ot = o.transpose()
	error = 1000000000000000000
	its = 0
	m = len(x)
	print('m %s' % m)
	its_hist = []
	err_hist = []
	xt = x.transpose()
	yt = y.transpose()
	while error > max_error:
		ot = update_o(x, xt, y, yt, m, ot)
		error = j(x, y, m, ot)
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

def admitted(o, x1, x2):
	return sigmoid(o.item(0) + o.item(1) * x1 + o.item(2) * x2)

def optimize_scipy(x, y, algorithm):
	x = np.array(x)
	y = np.array(y)
	mean_res, diff_res = normalize_x(x)
	o = np.zeros((3, 1))
	start = timer()
	Result = optimize.minimize(fun=j_scipy, x0=o, args=(x, y), method=algorithm, jac=j_o_der_scipy)
	end = timer()
	print(timedelta(seconds=end - start))
	o = Result.x
	print("o %s " % o)
	colors = ['green' if val else 'red' for val in y]
	plt.scatter(x[:, 1], x[:, 2], c=colors)
	line_x = [-0.5, 0.5]
	# because o1 + o2*x1 + 03*x2 = 0 for all the x1, x2 on such line
	line_y = - (o.item(0) + np.dot(o.item(1), line_x)) / o.item(2)
	plt.plot(line_x, line_y)
	x1 = 78
	x1 = (x1 - mean_res[0]) / diff_res[0]
	x2 = 78
	x2 = (x2 - mean_res[0]) / diff_res[0]
	print(admitted(o, x1, x2))
	plt.show()


# ------------        read data         ---------
x = []
y = []
with open('ex2data1.txt', 'r') as data_file:
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



# ------------        gm solution      ---------
# ------------        time elapsed: 0:00:34.703448s ------
# x = np.matrix(x)
# y = np.matrix(y)
# normalize_x(x)
# start = timer()
# o, its, its_hist, err_hist = gd(x, y)
# end = timer()
# print(timedelta(seconds=end-start))
# print('its %s' % its)
# print('o %s' % o)
# colors = ['green' if val else 'red' for val in y.A1]
# plt.scatter(x[:,1].A1, x[:,2].A1, c=colors)
# line_x = [-0.5, 0.5]
# # because o1 + o2*x1 + 03*x2 = 0 for all the x1, x2 on such line
# line_y = - (o.item(0) + np.dot(o.item(1), line_x)) / o.item(2)
# plt.plot(line_x, line_y)
# plt.show()
# ------------        gm solution         ---------


# ------------        scipy   BFGS      ---------
optimize_scipy(x, y, 'BFGS')
# ------------        scipy    BFGS     ---------


# ------------        scipy   Nelder-Mead      ---------
# optimize_scipy(x, y, 'Nelder-Mead')
# ------------        scipy    Nelder-Mead     ---------