import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

alpha = 0.001
max_error = 4.477

def h(o, o1, x):
	return o + o1*x

def j(target, m, o, o1):
	res = 0.0
	for i in range(m):
		x = target[0][i]
		y = target[1][i]
		res += (h(o, o1, x) - y) ** 2

	return res / (2*m)

def j_o_der(target, m, o, o1):
	res = 0.0
	for i in range(m):
		x = target[0][i]
		y = target[1][i]
		res += (h(o, o1, x) - y)

	return res / m

def j_o1_der(target, m, o, o1):
	res = 0.0
	for i in range(m):
		x = target[0][i]
		y = target[1][i]
		res += (h(o, o1, x) - y) * x
	return res / m

def gd(target):
	o = 0.0
	o1 = 0.0
	error = 1000000
	its = 0
	m = len(target[0])
	print('m %s' % m)
	while error > max_error:
		t_o = o - alpha * j_o_der(target, m, o, o1)
		t_o1 = o1 - alpha * j_o1_der(target, m, o, o1)
		o = t_o
		o1 = t_o1
		error = j(target, m, o, o1)
		its += 1

	print('its %s' % its)
	print('error %s' % error)
	return o, o1, its

data = [[], []]
with open('ex1data1.txt', 'r') as data_file:
	for line in data_file:
		data_i = [float(x) for x in line.split(',')]
		data[0].append(data_i[0])
		data[1].append(data_i[1])
print(data)

o, o1, its = gd(data)
print('its %s' % its)
print('o %s' % o)
print('o1 %s' % o1)

# ------     data only      -----------
#plt.plot(data[0], data[1], 'bo')
#plt.show()
# ------     data only      -----------

# ------     data + approximation      -----------
#x = np.linspace(0, 25, 1000)
#y = o + o1*x
#plt.plot(data[0], data[1], 'bo')
#plt.plot(x, y, '-r')
#plt.show()
# ------     data + approximation     -----------

# ------     j 3d      -----------
#fig = plt.figure()
#ax = plt.axes(projection="3d")

#X = np.arange(-4, 4, 0.01)
#Y = np.arange(-4, 4, 0.01)
#X, Y = np.meshgrid(X, Y)
#m = len(data[0])
#points_count = len(X)
#Z = np.array([j(data, m, X[i], Y[i]) for i in range(points_count)])
#ax.plot_wireframe(X, Y, Z, color='green')

#plt.show()
# ------     j 3d      -----------

# ------     j contour plot      -----------
fig, ax = plt.subplots()
X = np.arange(-4, -3.5, 0.01)
Y = np.arange(1, 1.5, 0.01)
X, Y = np.meshgrid(X, Y)
m = len(data[0])
points_count = len(X)
Z = np.array([j(data, m, X[i], Y[i]) for i in range(points_count)])
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=1, fontsize=10)
plt.show()
# ------     j contour plot      -----------