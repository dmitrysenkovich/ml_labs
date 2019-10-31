import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer
from datetime import timedelta

alpha = 0.001
max_error = 2043380050

def h(o, x):
	res = 0.0
	for i in range(len(x)):
		res += o[i] * x[i]

	return res

def j(x, y, m, o):
	res = 0.0
	for i in range(m):
		xi = x[i]
		yi = y[i]
		res += (h(o, xi) - yi) ** 2

	return res / (2 * m)

def j_oi_der(x, y, m, o, feature_i):
	res = 0.0
	for i in range(m):
		xi = x[i]
		yi = y[i]
		res += (h(o, xi) - yi) * xi[feature_i]

	return res / m

def update_o(x, y, m, o):
	o_len = len(o)
	new_o = [0] * o_len
	for i in range(o_len):
		j_oi_der_val = j_oi_der(x, y, m, o, i)
		new_o[i] = o[i] - alpha * j_oi_der_val

	return new_o

def normalize_x(x):
	dim = len(x[0])
	m = len(x)
	for dimi in range(1, dim):
		sum_dimi = 0.0
		min_dimi = float('inf')
		max_dimi = float('-inf')
		for i in range(m):
			sum_dimi += x[i][dimi]
			if x[i][dimi] < min_dimi:
				min_dimi = x[i][dimi]
			if x[i][dimi] > max_dimi:
				max_dimi = x[i][dimi]
		mean = sum_dimi / m
		for i in range(m):
			x[i][dimi] = (x[i][dimi] - mean) / (max_dimi - min_dimi)

def normalize_y(y):
	m = len(x)
	sum_y = sum(y)
	min_y = min(y)
	max_y = max(y)
	mean = sum_y / m
	for i in range(m):
		y[i] = (y[i] - mean) / (max_y - min_y)

def gd(x, y):
	o = [0.0, 0.0, 0.0]
	error = 1000000000000000000
	its = 0
	m = len(x)
	print('m %s' % m)
	its_hist = []
	err_hist = []
	while error > max_error:
		o = update_o(x, y, m, o)
		error = j(x, y, m, o)
		its += 1
		if (its % 100 == 0):
			its_hist.append(its)
			err_hist.append(error)
		if (its % 10000 == 0):
			print(error)

	print('its %s' % its)
	print('error %s' % error)
	return o, its, its_hist, err_hist


# ------------        gm vectorized         ---------

def h_vect(ot, x):
	return x * ot

def j_vect(x, yt, m, ot):
	h_vec_val = h_vect(ot, x)
	hy_diff = h_vec_val - yt
	return (hy_diff.transpose() * hy_diff)[0].item(0) / (2 * m)

def j_oi_der_vect(x, xt, y, yt, m, ot):
	h_vec_val = h_vect(ot, x)
	return (xt * (h_vec_val - yt)) / m

def update_o_vect(x, xt, y, yt, m, ot):
	j_oi_der_val = j_oi_der_vect(x, xt, y, yt, m, ot)
	return ot - alpha * j_oi_der_val

def gd_vect(x, y):
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
		ot = update_o_vect(x, xt, y, yt, m, ot)
		error = j_vect(x, yt, m, ot)
		its += 1
		if (its % 100 == 0):
			its_hist.append(its)
			err_hist.append(error)
		if (its % 10000 == 0):
			print(error)

	print('its %s' % its)
	print('error %s' % error)
	return ot, its, its_hist, err_hist

def normalize_x_np(x):
	dim = x[0].size
	m = len(x)
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
		for i in range(m):
			x[i].itemset(dimi, ((x[i].item(dimi) - mean) / (max_dimi - min_dimi)))


# ------------        gm vectorized         ---------



# ------------        read data         ---------
x = []
y = []
with open('ex1data2.txt', 'r') as data_file:
	for line in data_file:
		data_i = [float(x) for x in line.split(',')]
		x.append([1, data_i[0], data_i[1]])
		y.append(data_i[2])
# ------------        read data         ---------



# ------------        gm solution         ---------
# ------------        time elapsed: 0:00:34.703448s ------
#normalize_x(x)
#start = timer()
#o, its, its_hist, err_hist = gd(x, y)
#end = timer()
#print(timedelta(seconds=end-start))
#print('its %s' % its)
#print('o %s' % o)
# ------------        gm solution         ---------




# ------------        normal equation solution         ---------
#x_np = np.matrix(x)
#y_np = np.matrix(y)
#normalize_x_np(x_np)
#o = np.linalg.inv(x_np.transpose() * x_np) * (x_np.transpose()) * y_np.transpose()
#print(o)
# ------------        normal equation solution         ---------





# ------------        its/err         ---------
#plt.plot(its_hist, err_hist, 'bo')
#plt.show()
# ------------        its/err         ---------



# https://towardsdatascience.com/vectorization-implementation-in-machine-learning-ca652920c55d
# ------------        gm vectorized         ---------
# ------------        time elapsed: 0:00:25.165609s ------
#x_np = np.matrix(x)
#y_np = np.matrix(y)
#normalize_x_np(x_np)
#start = timer()
#o, its, its_hist, err_hist = gd_vect(x_np, y_np)
#end = timer()
#print(timedelta(seconds=end-start))
#print('its %s' % its)
#print('o %s' % o)
# ------------        gm vectorized         ---------

