import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import math
import scipy.optimize as optimize
import re

def calculate_error(x, theta, r, y):
	return np.multiply(x.dot(theta.T), r) - y

def cost(params, y, r, features_count, l):
	movies_count = y.shape[0]
	users_count = y.shape[1]
	x = np.reshape(params[:movies_count*features_count], (movies_count, features_count))
	theta = np.reshape(params[movies_count*features_count:], (users_count, features_count))

	error = 0.5*np.sum(calculate_error(x, theta, r, y)**2)
	penalty = 0.5*l*np.sum(x**2) + 0.5*l*np.sum(theta**2)

	return error + penalty

def gradient(params, y, r, features_count, l):
	movies_count = y.shape[0]
	users_count = y.shape[1]
	x = np.reshape(params[:movies_count*features_count], (movies_count, features_count))
	theta = np.reshape(params[movies_count*features_count:], (users_count, features_count))

	error = calculate_error(x, theta, r, y)
	x_gradient = error.dot(theta) + l*x
	theta_gradient = error.T.dot(x) + l*theta

	return np.concatenate([x_gradient.ravel(), theta_gradient.ravel()])

def normalize(y, r):
	m, n = y.shape
	y_mean = np.zeros((m, 1))
	y_normalized = np.zeros((m, n))
	for i in range(m):
		rating_indices = np.where(r[i, :] == 1)
		y_mean[i] = np.mean(y[i, rating_indices])
		y_normalized[i, rating_indices] = y[i, rating_indices] - y_mean[i]

	return y_normalized, y_mean

if __name__ == "__main__":



	# ------------        read data         ---------
	data = loadmat('ex9_movies.mat')
	y = data['Y']
	r = data['R']
	features_count = 3
	# ------------        read data         ---------



	# ------------        add my ratings         ---------
	my_y = np.zeros(shape=(y.shape[0], 1))
	my_y[226] = 5 # Star Trek VI: The Undiscovered Country
	my_y[227] = 4 # Star Trek: The Wrath of Khan
	my_y[228] = 5 # Star Trek III: The Search for Spock
	# check 229 Star Trek IV: The Voyage Home
	my_y[28] = 5 # Batman Forever
	my_y[230] = 5 # Batman Returns
	my_y[253] = 4 # Batman & Robin
	# check 402 Batman
	y = np.append(y, my_y, axis=1)
	my_r = np.zeros(shape=(y.shape[0], 1))
	my_r[226] = 1 # Star Trek VI: The Undiscovered Country
	my_r[227] = 1 # Star Trek: The Wrath of Khan
	my_r[228] = 1 # Star Trek III: The Search for Spock
	my_r[28] = 1 # Batman Forever
	my_r[230] = 1 # Batman Returns
	my_r[253] = 1 # Batman & Robin
	r = np.append(r, my_r, axis=1)
	# ------------        add my ratings         ---------



	# ------------        initialize parameters         ---------
	x = np.random.randn(y.shape[0], features_count)
	theta = np.random.randn(y.shape[1], features_count)
	params0 = np.concatenate([x.ravel(), theta.ravel()])
	print(cost(params0, y, r, features_count, 0.1))
	# ------------        initialize parameters         ---------



	# ------------        run optimization         ---------
	y_normalized, y_mean = normalize(y, r)
	optimal_params_unrolled = optimize.minimize(fun=cost, x0=params0, args=(y_normalized, r, features_count, 0.1), method='TNC', jac=gradient, options={'maxiter': 10000, 'disp': True})
	movies_count = y.shape[0]
	users_count = y.shape[1]
	optimal_x = np.reshape(optimal_params_unrolled.x[:movies_count * features_count], (movies_count, features_count))
	optimal_theta = np.reshape(optimal_params_unrolled.x[movies_count * features_count:], (users_count, features_count))
	# ------------        run optimization         ---------



	# ------------        my rating prediction         ---------
	my_theta = optimal_theta[943, :]
	star_track_x = optimal_x[229, :]
	batman_x = optimal_x[402, :]
	print('Star Trek IV: The Voyage Home rating prediction: %s' % star_track_x.dot(my_theta.T))
	print('Batman rating prediction: %s' % batman_x.dot(my_theta.T))
	# ------------        my rating prediction         ---------


	# ------------        my recommendations prediction         ---------
	movies = []
	with open('movie_ids.txt', 'r', encoding='iso-8859-1') as movies_file:
		for line in movies_file:
			movies.append(re.match(r"([0-9]+) ((.)+)", line, re.I).groups()[1])
	recommendations_count_to_show = 10
	p = optimal_x.dot(my_theta.T)
	my_rating_prediction = (p.reshape(-1, 1) + y_mean).flatten()
	max_ratings = my_rating_prediction.argsort()[::-1][:recommendations_count_to_show]
	print('My recommendations:')
	for movie_index in max_ratings:
		print('Movie: %s' % movies[movie_index])
	# ------------        my recommendations prediction         ---------
