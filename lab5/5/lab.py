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
import os
import email
from html import unescape
import functools
from collections import Counter
from random import shuffle
from scipy.io import savemat

special_characters = "!@#$%^&*()_+=-?/{[}]/*-+`~<>,.|\\:;\t\"\'―"
features_count = 2000

def extract_message_from_html(html):
	text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
	text = re.sub('<a\s.*?>', ' httpaddr ', text, flags=re.M | re.S | re.I)
	text = re.sub('<.*?>', '', text, flags=re.M | re.S)
	text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
	return unescape(text)

def extract_message(email):
	message = ""
	message_content_type = ""
	for part in email.walk():
		part_content_type = part.get_content_type()
		if part_content_type not in ("text/plain", "text/html"):
			continue

		message = str(part.get_payload())
		message_content_type = part_content_type
		if part_content_type == "text/plain":
			return message

	if message_content_type == "text/html":
		return extract_message_from_html(message)

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

def replace_new_lines(text):
	return text.replace('\n', ' ')

def clear_email(email):
	temp = email.lower()
	temp = remove_html_tags(temp)
	temp = replace_urls(temp)
	temp = replace_emails(temp)
	temp = replace_numbers(temp)
	temp = replace_dollar_signs(temp)
	temp = remove_special_characters(temp)
	temp = remove_multiple_spaces(temp)
	temp = replace_new_lines(temp)
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

	# ------------        read hams and spams         ---------
	# hams = []
	# for file_name in os.listdir("easy_ham"):
	# 	if file_name == 'cmds':
	# 		continue
	# 	with open('easy_ham/%s' % file_name, 'rb') as file:
	# 		msg = email.message_from_bytes(file.read())
	# 		hams.append(extract_message(msg))
	# spams = []
	# for file_name in os.listdir("spam"):
	# 	if file_name == 'cmds':
	# 		continue
	# 	with open('spam/%s' % file_name, 'rb') as file:
	# 		msg = email.message_from_bytes(file.read())
	# 		text_from_msg = extract_message(msg)
	# 		if text_from_msg:
	# 			spams.append(text_from_msg)
	# print('read emails')
	# ------------        read hams and spams         ---------




	# ------------        build vocabulary         ---------
	# cleared_hams = [clear_email(ham) for ham in hams]
	# cleared_spams = [clear_email(spam) for spam in spams]
	# print('cleared emails')
	#
	# ham_words = []
	# for cleared_ham in cleared_hams:
	# 	ham_words.extend([word for word in cleared_ham.split(' ') if len(word) > 2])
	# print('extracted ham words')
	# spam_words = []
	# for cleared_spam in cleared_spams:
	# 	spam_words.extend([word for word in cleared_spam.split(' ') if len(word) > 2])
	# print('extracted spam words')
	# all_words = ham_words + spam_words
	# most_common_words = Counter(all_words).most_common(features_count)
	# print('found most common words')
	#
	# with open('vocab.txt', 'w') as file:
	# 	for i in range(len(most_common_words)):
	# 		file.write("%s\t%s\n" % (i, most_common_words[i][0]))
	# print('built vocabulary')
	# ------------        build vocabulary         ---------



	# ------------        read vocabulary        ---------
	vocabulary = {}
	with open('vocab.txt', 'r') as vocabulary_file:
		for line in vocabulary_file:
			word_mapping = line.split('\t')
			vocabulary[word_mapping[1].replace('\n', '')] = word_mapping[0]
	# ------------        read vocabulary        ---------




	# ------------        build training and validation samples        ---------
	# shuffle(hams)
	# shuffle(spams)
	#
	# hams_train_count = 2000
	# spams_train_count = 400
	# emails_train_count = hams_train_count + spams_train_count
	# x = np.zeros(shape=(emails_train_count, features_count))
	# y = np.zeros(shape=(emails_train_count, 1))
	# for i in range(emails_train_count):
	# 	if i < hams_train_count:
	# 		ham_feature_vector = clear_and_map_email_to_feature_vector(hams[i], vocabulary)
	# 		x[i] = ham_feature_vector
	# 		y[i] = 0
	# 	else:
	# 		spam_feature_vector = clear_and_map_email_to_feature_vector(spams[i - hams_train_count], vocabulary)
	# 		x[i] = spam_feature_vector
	# 		y[i] = 1
	# print('built training set')
	#
	# hams_validation_count = 500
	# spams_validation_count = 99
	# emails_validation_count = hams_validation_count + spams_validation_count
	# x_val = np.zeros(shape=(emails_validation_count, features_count))
	# y_val = np.zeros(shape=(emails_validation_count, 1))
	# for i in range(emails_validation_count):
	# 	if i < hams_validation_count:
	# 		ham_index = hams_train_count + i
	# 		ham_feature_vector = clear_and_map_email_to_feature_vector(hams[ham_index], vocabulary)
	# 		x_val[i] = ham_feature_vector
	# 		y_val[i] = 0
	# 	else:
	# 		spam_index = spams_train_count + i - hams_validation_count
	# 		spam_feature_vector = clear_and_map_email_to_feature_vector(spams[spam_index], vocabulary)
	# 		x_val[i] = spam_feature_vector
	# 		y_val[i] = 1
	# print('built validation set')
	#
	# savemat('spamTrain.mat', {'X': x, 'y': y})
	# savemat('spamTest.mat', {'Xtest': x_val, 'ytest': y_val})
	# print('saved train and validation samples')
	# ------------        build training and validation samples        ---------





	# ------------        read data         ---------
	data = loadmat("spamTrain.mat")
	x = data['X']
	y = data['y']
	data_test = loadmat("spamTest.mat")
	x_test = data_test['Xtest']
	y_test = data_test['ytest']
	# ------------        read data         ---------



	# ------------        searching for the best C and sigma squared        ---------
	# number_of_C_samples = 5
	# number_of_gamma_samples = 20
	# C_to_check = np.linspace(2, 4, number_of_C_samples)
	# gamma_to_check = np.linspace(0.001, 0.02, number_of_gamma_samples)
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
	# best_C 3.0, best_sigma_squared 62.5, best_score 0.989983305509182
	# ------------        searching for the best C and sigma squared        ---------



	# ------------        best parameters score         ---------
	best_C = 3.0
	best_sigma_squared = 62.5
	gamma = 1 / (2*best_sigma_squared)
	svclassifier = SVC(C=best_C, gamma=gamma, kernel='rbf')
	svclassifier.fit(x, y.flatten())
	score = svclassifier.score(x_test, y_test.flatten())
	print(score)
	# ------------        best parameters score         ---------




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
