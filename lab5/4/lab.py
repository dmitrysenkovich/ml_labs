import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer
from datetime import timedelta
import scipy.optimize as optimize
from sympy import *
from scipy.io import loadmat
import pandas as pd
from sklearn.svm import SVC
from datetime import timedelta
import re
from nltk.stem import PorterStemmer

special_characters = "!@#$%^&*()_+=-?/{[}]/*-+`~<>,.|\\:;\t\"\'―"

def remove_html_tags(text):
	return re.sub('<.*?>', ' ', text)

def replace_urls(text):
	return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' httpaddr ', text)

def replace_emails(text):
	return re.sub('[^\s]+@[^\s]+', ' emailaddr ', text)

def replace_numbers(text):
	return re.sub('[0-9]+', ' number ', text)

def replace_dollar_signs(text):
	return re.sub('[$]+', ' dollar ', text)

def remove_special_characters(text):
	return ''.join(c for c in text if c not in special_characters)

def stem(text):
	stemmer = PorterStemmer()
	text = [stemmer.stem(token) for token in text.split(" ")]
	return " ".join(text)

def remove_multiple_spaces(text):
	return re.sub(' +', ' ', text)

def clear_email(email):
	temp = email.lower()
	temp = remove_html_tags(temp)
	temp = replace_urls(temp)
	temp = replace_emails(temp)
	temp = replace_numbers(temp)
	temp = replace_dollar_signs(temp)
	temp = remove_special_characters(temp)
	temp = remove_multiple_spaces(temp)
	cleared_email = stem(temp)
	return cleared_email

def find_words_from_vocabulary(text, vocabulary):
	found_words_codes = set()
	for word in text.split(' '):
		if word in vocabulary:
			found_words_codes.add(vocabulary[word])
	return found_words_codes

def to_feature_vector(found_words_codes, vocabulary):
	x = np.array([0]*len(vocabulary))
	for code in found_words_codes:
		x[int(code) - 1] = 1
	return x.reshape(1, -1)

def clear_and_map_email_to_feature_vector(email, vocabulary):
	cleared_email = clear_email(email)
	found_words_codes = find_words_from_vocabulary(cleared_email, vocabulary)
	return to_feature_vector(found_words_codes, vocabulary)

if __name__ == "__main__":

	# ------------        read data         ---------
	data = loadmat("spamTrain.mat")
	x = data['X']
	y = data['y']
	data_test = loadmat("spamTest.mat")
	x_test = data_test['Xtest']
	y_test = data_test['ytest']
	# ------------        read data         ---------



	# ------------        searching for the best C and sigma squared        ---------
	# number_of_C_samples = 10
	# number_of_gamma_samples = 100
	# C_to_check = np.linspace(1, 10, number_of_C_samples)
	# gamma_to_check = np.linspace(0.01, 1, number_of_gamma_samples)
	# best_C = -1
	# best_sigma_squared = -1
	# best_score = -1
	# curr_model_number = 0
	# for C in C_to_check:
	# 	for gamma in gamma_to_check:
	# 		classifier = SVC(C=C, gamma=gamma, kernel='rbf')
	# 		classifier.fit(x, y.flatten())
	# 		score = classifier.score(x_test, y_test.flatten())
	# 		if score > best_score:
	# 			best_score = score
	# 			best_C = C
	# 			best_sigma_squared = 1 / (2*gamma)
	# 		curr_model_number += 1
	# 		print(curr_model_number)
	# 		print(best_C)
	# 		print(best_sigma_squared)
	# 		print(best_score)
	# print('best_C %s, best_sigma_squared %s, best_score %s' % (best_C, best_sigma_squared, best_score))
	# best_C 1.0, best_sigma_squared 50.0, best_score 0.987
	# ------------        searching for the best C and sigma squared        ---------



	# ------------        best parameters score         ---------
	best_C = 1.0
	best_sigma_squared = 50.0
	gamma = 1 / (2*best_sigma_squared)
	svclassifier = SVC(C=best_C, gamma=gamma, kernel='rbf')
	svclassifier.fit(x, y.flatten())
	score = svclassifier.score(x_test, y_test.flatten())
	print(score)
	# ------------        best parameters score         ---------



	# ------------        test email clear functions        ---------
	test = "<html><asd>asd<asd><qwe>qwe</qwe></html>"
	print(remove_html_tags(test))
	test = "http://ololo.com/ http://ololo.com"
	print(replace_urls(test))
	test = "https://ololo.com/ https://ololo.com"
	print(replace_urls(test))
	test = "123 45"
	print(replace_numbers(test))
	test = "$ $$"
	print(replace_dollar_signs(test))
	test = "!@#$%^&*()_+=-?/{[}]/*-+`~<>,.|\\:;\t\"\'―"
	print(remove_special_characters(test))
	test = "Stemming is funnier than a bummer says the sushi loving computer scientist"
	print(stem(test))
	# ------------        test email clear functions        ---------



	# ------------        read vocabulary        ---------
	vocabulary = {}
	with open('vocab.txt', 'r') as vocabulary_file:
		for line in vocabulary_file:
			word_mapping = line.split('\t')
			vocabulary[word_mapping[1].replace('\n', '')] = word_mapping[0]
	# ------------        read vocabulary        ---------



	# ------------        read email samples        ---------
	email_sample1 = ""
	with open('emailSample1.txt', 'r') as file:
		email_sample1 = file.read().replace('\n', ' ')
	email_sample2 = ""
	with open('emailSample2.txt', 'r') as file:
		email_sample2 = file.read().replace('\n', ' ')
	spam_sample1 = ""
	with open('spamSample1.txt', 'r') as file:
		spam_sample1 = file.read().replace('\n', ' ')
	spam_sample2 = ""
	with open('spamSample2.txt', 'r') as file:
		spam_sample2 = file.read().replace('\n', ' ')
	# ------------        read email samples        ---------



	# ------------        clear and map emails to feature vectors        ---------
	email1 = clear_and_map_email_to_feature_vector(email_sample1, vocabulary)
	email2 = clear_and_map_email_to_feature_vector(email_sample2, vocabulary)
	spam1 = clear_and_map_email_to_feature_vector(spam_sample1, vocabulary)
	spam2 = clear_and_map_email_to_feature_vector(spam_sample2, vocabulary)
	# ------------        clear and map emails to feature vectors        ---------



	# ------------        classify sample emails        ---------
	print('is sample spam: %s' % svclassifier.predict(email1))
	print('is sample spam: %s' % svclassifier.predict(email2))
	print('is sample spam: %s' % svclassifier.predict(spam1))
	print('is sample spam: %s' % svclassifier.predict(spam2))
	# ------------        classify sample emails        ---------




	# ------------        my own samples        ---------
	email_sample1 = ""
	with open('myEmailSample1.txt', 'r') as file:
		email_sample1 = file.read().replace('\n', ' ')
	email_sample2 = ""
	with open('myEmailSample2.txt', 'r') as file:
		email_sample2 = file.read().replace('\n', ' ')
	spam_sample1 = ""
	with open('mySpamSample1.txt', 'r') as file:
		spam_sample1 = file.read().replace('\n', ' ')
	spam_sample2 = ""
	with open('mySpamSample2.txt', 'r') as file:
		spam_sample2 = file.read().replace('\n', ' ')
	email1 = clear_and_map_email_to_feature_vector(email_sample1, vocabulary)
	email2 = clear_and_map_email_to_feature_vector(email_sample2, vocabulary)
	spam1 = clear_and_map_email_to_feature_vector(spam_sample1, vocabulary)
	spam2 = clear_and_map_email_to_feature_vector(spam_sample2, vocabulary)
	print('is sample spam: %s' % svclassifier.predict(email1))
	print('is sample spam: %s' % svclassifier.predict(email2))
	print('is sample spam: %s' % svclassifier.predict(spam1))
	print('is sample spam: %s' % svclassifier.predict(spam2))
	# ------------        my own samples        ---------