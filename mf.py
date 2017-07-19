# coding:utf-8

import numpy as np
import pandas as pd

# step1----------read dataset
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/u.data', sep = '\t', names = header)

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

# step3=============sparsity
sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
print str(sparsity * 100) + '%'

# step4==============mf
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
	prediction = prediction[ground_truth.nonzero()].flatten()
	ground_truth = ground_truth[ground_truth.nonzero()].flatten()
	return sqrt(mean_squared_error(prediction, ground_truth))

import scipy.sparse as sp
from scipy.sparse.linalg import svds

u,s,vt = svds(train_data_matrix, k = 20)
s_diag_matrix = np.diag(s)
x_pred = np.dot(np.dot(u, s_diag_matrix), vt)

print 'MSE:'+ str(rmse(x_pred, test_data_matrix))