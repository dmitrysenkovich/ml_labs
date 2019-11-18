import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import math

def p(x, mu, sigma):
	sigma = np.diag(sigma.ravel())
	sigma_det = np.linalg.det(sigma)
	n = mu.shape[1]
	first_part = (((2*math.pi)**(n/2))*(sigma_det**0.5))**-1

	sigma_inv = np.linalg.pinv(sigma)
	diff = x - mu
	second_part = np.exp(-0.5 * np.sum(diff.dot(sigma_inv) * diff, 1)).reshape(-1, 1)

	return first_part * second_part

def f_score(res):
	predicted_positive_total = res[0][0] + res[0][1]
	precision = res[0][0] / (predicted_positive_total if predicted_positive_total > 0 else 1)
	actual_positive_total = res[0][0] + res[1][0]
	recall = res[0][0] / (actual_positive_total if actual_positive_total > 0 else 1)
	return 2*precision*recall/((precision + recall) if (precision + recall) > 0 else 1)

def estimate(x):
	mu = np.array([np.mean(x[:, i]) for i in range(x.shape[1])]).reshape(1, -1)
	sigma = compute_sigma(x, mu)

	return mu, sigma

def compute_sigma(x, mu):
	m = x.shape[0]
	features_count = x.shape[1]
	sum = np.zeros(shape=(1, features_count))
	for xi in x:
		diff = xi - mu
		sum += diff ** 2

	return sum / m

def find_best_epsilon(x_val, y_val, mu, sigma):
	p_vals = p(x_val, mu, sigma)
	step = (np.max(p_vals) - np.min(p_vals)) / 1000
	epsilons = np.arange(np.min(p_vals), np.max(p_vals), step)
	best_f_score = 0
	best_epsilon = -1
	best_confusion_matrix = []
	for epsilon in epsilons:
		true_positives = 0.0
		false_positives = 0.0
		false_negatives = 0.0
		true_negatives = 0.0
		is_outliers = p_vals < epsilon
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
		#plt.scatter([*range(1, len(x_val) + 1)], p_vals)
		#plt.show()
		if score > best_f_score:
			best_f_score = score
			best_epsilon = epsilon
			best_confusion_matrix = np.array([
				[true_positives, false_positives],
				[false_negatives, true_negatives]
			])
			print(epsilon)
			print(best_f_score)
			print(best_confusion_matrix)
	return best_epsilon, best_f_score, best_confusion_matrix

if __name__ == "__main__":



	# ------------        read data         ---------
	data = loadmat('ex8data2.mat')
	x = data['X']
	x_val = data['Xval']
	y_val = data['yval']
	# ------------        read data         ---------



	# ------------        estimate x         ---------
	mu, sigma = estimate(x)
	# ------------        estimate x         ---------


	# ------------        find best epsilon         ---------
	best_epsilon, best_f_score, best_confusion_matrix = find_best_epsilon(x_val, y_val, mu, sigma)
	print('best f score: %s' % best_f_score)
	print('best epsilon: %s' % best_epsilon)
	# ------------        find best epsilon         ---------


	# ------------        anomalies in the training set         ---------
	p_vals = p(x, mu, sigma)
	anomalies_count = sum(p_vals < best_epsilon)
	print('anomalies_count: %s' % anomalies_count)
	# ------------        anomalies in the training set         ---------
