import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def read_data(lines_to_read = 1000):
    data_file_name = "data/" + "Base128.txt"

    x = []
    y = []
    with open(data_file_name) as file:
        lines = file.readlines()[:lines_to_read]
        for line in lines:
            xi, yi = line.split()
            challenge = [int(character) for character in xi]
            x.append(challenge)
            y.append(int(yi))

    return np.array(x), np.array(y)


if __name__ == "__main__":



    # ------------        read data         ---------
    all_x, all_y = read_data()
    x, x_val, y, y_val = train_test_split(all_x, all_y, train_size = 0.8, test_size = 0.2)
    print('done reading data')
    # ------------        read data         ---------




    # ------------        first model         ---------
    # number_of_C_samples = 10
    # number_of_gamma_samples = 10
    # C_to_check = np.linspace(0.1, 1, number_of_C_samples)
    # gamma_to_check = np.linspace(0.0001, 0.001, number_of_gamma_samples)
    # best_C = -1
    # best_gamma = -1
    # best_score = -1
    # curr_model_number = 0
    # for C in C_to_check:
    #     for gamma in gamma_to_check:
    #         classifier = SVC(C = C, gamma = gamma, kernel='rbf')
    #         classifier.fit(x, y.flatten())
    #         score = classifier.score(x_val, y_val.flatten())
    #         if score > best_score:
    #             best_score = score
    #             best_C = C
    #             best_gamma = gamma
    #             print('------')
    #             print('best score so far: %s' % best_score)
    #             print(best_C)
    #             print(best_gamma)
    #             print('------')
    #         curr_model_number += 1
    #         if curr_model_number % 1000 == 0:
    #             print(curr_model_number)
    # print('best_C %s, best_gamma %s, best_score %s' % (best_C, best_gamma, best_score))
    # best_C 0.1, best_gamma 0.0001, best_score 0.665
    classifier = SVC(C = 0.1, gamma = 0.0001, kernel='rbf')
    classifier.fit(x, y.flatten())
    predictions = classifier.predict(x_val)
    print(accuracy_score(y_val, predictions))
    # ------------        first model         ---------



    # ------------        3 different algorithms         ---------
    # 0.67 0.65 0.67
    # classifiers = [SVC(C = 0.1, gamma = 0.0001, kernel='rbf'), GradientBoostingClassifier(n_estimators=500, max_depth=5), QuadraticDiscriminantAnalysis()]
    # for classifier in classifiers:
    #     classifier.fit(x, y)
    #     predictions = classifier.predict(x_val)
    #     print(accuracy_score(y_val, predictions))
    # ------------        3 different algorithms         ---------




    # ------------        accuracy depending on samples count         ---------
    samples_counts = []
    accuracies = []
    for samples_count in range(10, 800, 10):
        xi = x[:samples_count]
        yi = y[:samples_count]
        xi_val = x_val[:samples_count]
        yi_val = y_val[:samples_count]
        classifier = GradientBoostingClassifier()
        classifier.fit(x, y)
        predictions = classifier.predict(xi_val)
        samples_counts.append(samples_count)
        accuracies.append(accuracy_score(yi_val, predictions))
    plt.plot(samples_counts, accuracies)
    plt.show()
    # ------------        accuracy depending on samples count         ---------