# coding:utf-8
import numpy as np
import pandas as pd

# step1----------read dataset
header = ['user_id', 'item_id', 'rating', 'timestamp']
df= pd.read_csv('./ml-100k/u.data', sep = '\t', names = header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

print n_users, n_items

# step2----------make train & test dataset
from sklearn import model_selection as cv
train_data, test_data = cv.train_test_split(df, test_size = 0.25)

train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
	train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
	test_data_matrix[line[1]-1, line[2]-1] = line[3]

# step3-----------similarity matrix
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric = 'cosine')
item_similarity = pairwise_distances(train_data_matrix, metric = 'cosine')

# step4-----------make prediction
def predict(rating, similarity, type = 'item'):
	if type == 'user':
		mean_user_rating = rating.mean(axis = 1)
		rating_diff = (rating - mean_user_rating[:,np.newaxis])
		pred = mean_user_rating[:,np.newaxis] + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
	elif type == 'item':
		pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
	return pred

user_prediction = predict(train_data_matrix, user_similarity, type = 'user')
# item_prediction = predict(train_data_matrix, item_similarity, type = 'item')

# step5-----------RMSE
from sklearn.metrics import mean_squared_error

from math import sqrt

def rmse(prediction, ground_truth):
	prediction = prediction[ground_truth.nonzero()].flatten()
	ground_truth = ground_truth[ground_truth.nonzero()].flatten()
	return sqrt(mean_squared_error(prediction, ground_truth))

print 'User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
# print 'Item based CF RMSe: ' + str(rmse(item_prediction, test_data_matrix))