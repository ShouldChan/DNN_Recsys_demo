# coding:utf-8

import numpy as np
import pandas as pd

# step1----------read dataset
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./u.data', sep = '\t', names = header)

users = df.user_id.unique()
items = df.item_id.unique()

n_users = users.shape[0]
n_items = items.shape[0]

# print type(users)
# print users
# print items
print n_users, n_items

# step1.5------------map userid/itemid to matrix_index
uid_2_uindex = {}
ucount = 0
iid_2_iindex = {}
icount = 0

for i in range(n_users):
	uid_2_uindex[users[i]] = ucount
	ucount += 1

for j in range(n_items):
	iid_2_iindex[items[j]] = icount
	icount += 1

# print sorted(uid_2_uindex.iteritems(), key = lambda asd:asd[1], reverse = False)
# print iid_2_iindex


# step2----------make train & test dataset
from sklearn import model_selection as cv
train_data, test_data = cv.train_test_split(df, test_size = 0.25)

train_data_matrix = np.zeros((n_users, n_items))
print type(train_data_matrix)
for line in train_data.itertuples():
	# print line[1],line[2]
	# print uid_2_uindex[line[1]], iid_2_iindex[line[2]], line[3]
	train_data_matrix[uid_2_uindex[line[1]], iid_2_iindex[line[2]]] = float(line[3])


test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
	test_data_matrix[uid_2_uindex[line[1]], iid_2_iindex[line[2]]] = float(line[3])

# step3=============sparsity
sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
print str(sparsity * 100) + '%'

# # step4==============mf
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

print 'RMSE:'+ str(rmse(x_pred, test_data_matrix))