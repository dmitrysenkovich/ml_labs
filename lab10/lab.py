from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree, metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def predict(x, trained_trees, weights):
	return [sum([weight * tree.predict([xi])[0] for tree, weight in zip(trained_trees, weights)]) for xi in x]


def gradient_boosting(x, y, trees_count, max_depth, weights):
	trained_trees = []
	curr_labels = np.array(y)
	for _ in range(trees_count):
		trained_tree = tree.DecisionTreeRegressor(max_depth = max_depth, random_state = 42).fit(x, curr_labels)
		trained_trees.append(trained_tree)
		curr_labels = y - predict(x, trained_trees, weights)
	return trained_trees


def rmse(x, y, trained_trees, weights):
	return np.sqrt(metrics.mean_squared_error(y, predict(x, trained_trees, weights)))


def rmse_sklearn(x, y, predictor):
	return np.sqrt(metrics.mean_squared_error(y, predictor.predict(x)))


if __name__ == "__main__":



	# ------------        read data         ---------
	data = load_boston()
	X = data.data
	Y = data.target
	training_data_set_length = int(X.shape[0]*0.75)
	x = X[:training_data_set_length]
	y = Y[:training_data_set_length]
	x_test = X[training_data_set_length:]
	y_test = Y[training_data_set_length:]
	# ------------        read data         ---------



	# ------------        create trees and predict test samples results          ---------
	trees_count = 50
	max_depth = 5
	weights = [0.9] * trees_count
	trained_trees = gradient_boosting(x, y, trees_count, max_depth, weights)
	print('predictions:')
	print(predict(x_test, trained_trees, weights))
	print('rmse: %s' % rmse(x_test, y_test, trained_trees, weights))
	# rmse: 5.455565103009402
	# ------------        create trees and predict test samples results          ---------



	# ------------        create trees and predict test samples results varying weights          ---------
	trees_count = 50
	max_depth = 5
	weights = [(0.9 / (1.0 + i)) for i in range(trees_count)]
	trained_trees = gradient_boosting(x, y, trees_count, max_depth, weights)
	print('predictions:')
	print(predict(x_test, trained_trees, weights))
	print('rmse: %s' % rmse(x_test, y_test, trained_trees, weights))
	# rmse: 4.812550945781193
	# ------------        create trees and predict test samples results varying weights          ---------




	# ------------        investigate results with different parameters          ---------
	trees_counts = np.linspace(10, 100, 10)
	max_depths = np.linspace(2, 20, 10)
	fig, ax = plt.subplots(nrows = 2, ncols = 5)
	ax = ax.flatten()
	i = 0
	for max_depth in max_depths:
		rmse_train = []
		rmse_test = []
		for trees_count in trees_counts:
			gradient_boosting_regressor = GradientBoostingRegressor(n_estimators = int(trees_count), max_depth = int(max_depth), random_state = 42).fit(x, y)
			rmse_train.append(rmse_sklearn(x, y, gradient_boosting_regressor))
			rmse_test.append(rmse_sklearn(x_test, y_test, gradient_boosting_regressor))
		ax[i].plot(trees_counts, rmse_train, color = "green")
		ax[i].plot(trees_counts, rmse_test, color = "red")
		i += 1
	plt.show()
	# ------------        investigate results with different parameters          ---------




	# ------------        compare with linear regression          ---------
	linear_regression = LinearRegression().fit(x, y)
	print('linear regression rmse: %s' % rmse_sklearn(x_test, y_test, linear_regression))
	# linear regression rmse: 8.254979753549401
	# ------------        compare with linear regression          ---------
