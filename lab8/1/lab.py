import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import math

def plot_data(x):
	plt.scatter(x[:, 0], x[:, 1])

def plot_histogram(x):
	plt.hist(x, bins=70)

def p(x, means, stds):
	probability = 1.0
	for i in range(x.shape[0]):
		mean = means[i]
		std = stds[i]
		xi = x[i]
		probability *= (1.0/(math.sqrt(2*math.pi)*std))*np.exp(-((xi - mean)**2)/(2*(std**2)))
	return probability

def is_outlier(x, means, stds, epsilon):
	return p(x, means, stds) < epsilon

def f_score(res):
	predicted_positive_total = res[0][0] + res[0][1]
	precision = res[0][0] / (predicted_positive_total if predicted_positive_total > 0 else 1)
	actual_positive_total = res[0][0] + res[1][0]
	recall = res[0][0] / (actual_positive_total if actual_positive_total > 0 else 1)
	return 2*precision*recall/((precision + recall) if (precision + recall) > 0 else 1)

def plot_distribution(means, stds):
	x1, x2 = np.meshgrid(np.arange(0, 30, 0.1), np.arange(0, 30, 0.1))
	points_count = len(x1)
	z = np.zeros(shape=(points_count, points_count))
	for i in range(points_count):
		for j in range(points_count):
			z[i, j] = p(np.array([x1[i, i], x2[j, j]]).reshape(-1, 1), means, stds)
	levels = [10 ** exp for exp in range(-60, 0, 3)]
	plt.contour(x1, x2, z, levels=levels)

def estimate(x):
	means = [np.mean(x[:, i]) for i in range(x.shape[1])]
	stds = [np.std(x[:, i]) for i in range(x.shape[1])]
	return means, stds

def find_best_epsilon(x_val, y_val, means, stds):
	best_f_score = 0
	best_epsilon = -1
	p_values = [p(x_val[i], means, stds) for i in range(len(x_val))]
	step = (np.max(p_values) - np.min(p_values)) / 1000
	epsilons = np.arange(np.min(p_values), np.max(p_values), step)
	for epsilon in epsilons:
		true_positives = 0.0
		false_positives = 0.0
		false_negatives = 0.0
		true_negatives = 0.0
		is_outliers = p_values < epsilon
		for i in range(len(x_val)):
			yi = y_val[i, 0]
			is_positive = 1 if is_outliers[i] else 0
			if is_positive == 1 and yi == 1:
				true_positives += 1
			elif is_positive == 1 and yi == 0:
				false_positives += 1
			elif is_positive == 0 and yi == 1:
				false_negatives += 1
			elif is_positive == 0 and yi == 0:
				true_negatives += 1
		score = f_score(np.array([
			[true_positives, false_positives],
			[false_negatives, true_negatives]
		]))
		if score > best_f_score:
			best_f_score = score
			best_epsilon = epsilon
			print(best_f_score)
			print(np.array([
				[true_positives, false_positives],
				[false_negatives, true_negatives]
			]))
	return best_epsilon, best_f_score

def plot_outliers(x, means, stds, epsilon):
	for xi in x:
		if is_outlier(xi, means, stds, epsilon):
			plt.plot(xi[0], xi[1], 'ro', c='r')

if __name__ == "__main__":



	# ------------        read data         ---------
	data = loadmat('ex8data1.mat')
	x = data['X']
	x_val = data['Xval']
	y_val = data['yval']
	# ------------        read data         ---------


	# ------------        plot data         ---------
	# plot_data(x)
	# plt.show()
	# ------------        plot data         ---------


	# ------------        plot histograms         ---------
	# plot_histogram(x[:, 0])
	# plt.show()
	# plot_histogram(x[:, 1])
	# plt.show()
	# ------------        plot histograms         ---------



	# ------------        estimate x         ---------
	means, stds = estimate(x)
# ------------        estimate x         ---------



	# ------------        plot distribution with data         ---------
	# plot_data(x)
	# plot_distribution(means, stds)
	# plt.show()
	# ------------        plot distribution with data         ---------


	# ------------        find best epsilon         ---------
	best_epsilon, best_f_score = find_best_epsilon(x_val, y_val, means, stds)
	print('best f score: %s' % best_f_score)
	print('best epsilon: %s' % best_epsilon)
	# ------------        find best epsilon         ---------



	# ------------        plot data and outliers         ---------
	plot_data(x)
	plot_distribution(means, stds)
	plot_outliers(x, means, stds, best_epsilon)
	plt.show()
	# ------------        plot data and outliers         ---------
