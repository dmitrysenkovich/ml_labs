import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC

def plot_data(x, y):
	colors = ['green' if val else 'red' for val in y]
	plt.scatter(x[:, 0], x[:, 1], c=colors)

def plot_nonlinear_decision_boundary(svclassifier, x, y):
	x_min, x_max = x[:, 0].min(), x[:, 0].max()
	y_min, y_max = x[:, 1].min(), x[:, 1].max()
	ax = np.linspace(x_min, x_max, 1000)
	ay = np.linspace(y_min, y_max, 1000)
	xx = np.array([(axx, ayy) for axx in ax for ayy in ay])

	xx_predictions = svclassifier.predict(xx)
	colors = ['#90ee90' if prediction else '#ff9185' for prediction in xx_predictions]
	plt.scatter(xx[:, 0], xx[:, 1], c=colors, s = 5)

	plot_data(x, y)

	plt.show()

if __name__ == "__main__":

	# ------------        read data         ---------
	data = loadmat("ex5data3.mat")
	x = data['X']
	y = data['y']
	x_val = data['Xval']
	y_val = data['yval']
	# ------------        read data         ---------



	# ------------        plot data         ---------
	# plot_data(x_val, y_val)
	# plt.show()
	# ------------        plot data         ---------



	# ------------        searching for the best C and sigma squared        ---------
	# number_of_C_samples = 100
	# number_of_gamma_samples = 100
	# C_to_check = np.linspace(0.1, 100, number_of_C_samples)
	# gamma_to_check = np.linspace(0.01, 1, number_of_gamma_samples)
	# best_C = -1
	# best_sigma_squared = -1
	# best_score = -1
	# curr_model_number = 0
	# for C in C_to_check:
	# 	for gamma in gamma_to_check:
	# 		classifier = SVC(C=C, gamma=gamma, kernel='rbf')
	# 		classifier.fit(x, y.flatten())
	# 		score = classifier.score(x_val, y_val.flatten())
	# 		if score > best_score:
	# 			best_score = score
	# 			best_C = C
	# 			best_sigma_squared = 1 / (2*gamma)
	# 		curr_model_number += 1
	# 		if curr_model_number % 1000 == 0:
	# 			print(curr_model_number)
	# print('best_C %s, best_sigma_squared %s, best_score %s' % (best_C, best_sigma_squared, best_score))
	# best_C 1.1090909090909091, best_sigma_squared 0.6329113924050632, best_score 0.945
	# ------------        searching for the best C and sigma squared        ---------



	# ------------        validation sample and approximation with best parameters         ---------
	best_C = 1.1090909090909091
	best_sigma_squared = 0.6329113924050632
	gamma = 1 / (2*best_sigma_squared)
	svclassifier = SVC(C=best_C, gamma=gamma, kernel='rbf')
	svclassifier.fit(x, y.flatten())
	plot_nonlinear_decision_boundary(svclassifier, x_val, y_val)
	# ------------        validation sample and approximation with best parameters         ---------

