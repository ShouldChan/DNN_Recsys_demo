# coding:utf-8

import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import time

def read_dataset():
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
    return df, users, items, n_users, n_items

def map_id2index(users, items, n_users, n_items):
    # step2------------map userid/itemid to matrix_index
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
    return uid_2_uindex, iid_2_iindex

def make_traintest(df, uid_2_uindex, iid_2_iindex):
    # step3----------make train & test dataset
    from sklearn import model_selection as cv
    train_data, test_data = cv.train_test_split(df, test_size = 0.25)

    train_data_matrix = np.zeros((n_users, n_items))
    # print type(train_data_matrix)
    for line in train_data.itertuples():
        # print line[1],line[2]
        # print uid_2_uindex[line[1]], iid_2_iindex[line[2]], line[3]
        train_data_matrix[uid_2_uindex[line[1]], iid_2_iindex[line[2]]] = float(line[3])


    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[uid_2_uindex[line[1]], iid_2_iindex[line[2]]] = float(line[3])

    return train_data_matrix, test_data_matrix 

def calc_sparsity(df, n_users, n_items):
    # step4=============sparsity
    sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
    spar_2_str = str(sparsity * 100) + '%'
    return spar_2_str

# mse
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

# dnn+pmf
def DNNPMF(ratings, n_factors=40, learning_rate=0.01, _lambda_1=0.01, _lambda_2=0.01, alpha=0.01):
	# ratings: ndarray
	# 1.define an indicator matrix I, I_ij = 1 if R_ij > 0 and 0 otherwise
	I_indicator = ratings
	I_indicator[I_indicator>0] = 1
	I_indicator[I_indicator<=0] = 0
	# print I_indicator

	# 2.learning == 'sgd'
	# nonzero返回非零元素的索引值数组 
	# 二维数组的话第一个array从行角度描述其索引值，第二个从列
	sample_row, sample_col = ratings.nonzero()
 	n_samples = len(sample_row) # 非零元素的个数 为正例
 	# print sample_row, sample_col
 	# print n_samples

 	# 3.initialize U V
 	n_users, n_items=ratings.shape
 	# print n_users, n_items
 	user_vecs = np.random.normal(scale=1./n_factors, \
 		size=(n_users, n_factors))
 	item_vecs = np.random.normal(scale=1./n_factors, \
 		size=(n_items, n_factors))
 	# print user_vecs, item_vecs
 	# (2113, 40) (8887, 40)
 	print user_vecs.shape, item_vecs.shape

 	# 4.



if __name__ == "__main__":
    # step1-------read dataset
    t = time.time()
    df, users, items, n_users, n_items = read_dataset()
    print 'step1--read dataset\telapse:', time.time() - t

    # step2----------map userid/itemid to matrix_index
    uid_2_uindex, iid_2_iindex = map_id2index(users, items, n_users, n_items)
    print 'step2--map userid/itemid to matrix index\telapse:', time.time() - t

    # step3----------make train and test dataset
    train_data_matrix, test_data_matrix  = make_traintest(df, uid_2_uindex, iid_2_iindex)
    print 'step3--make train and test dataset\telapse:', time.time() - t
    print type(train_data_matrix)

    # step4---------calculate sparsity
    print calc_sparsity(df, n_users, n_items)
    print 'step4--calculate sparsity\telapse:', time.time() - t

    # test5
    # poster_path = './1.jpg'
    # feature_vec = get_picfeature(poster_path)
    # print '-----'
    


    # step5---------use vgg16 to extract the pic features
    # with open('./movie_ID_Jpg.txt', 'rb') as fopen:
    #     lines = fopen.readlines()
    #     for line in lines:
    #         tempdata = line.strip().split('\t')
    #         movie_id, movie_jpg = tempdata[0], tempdata[1]
            
    #         print str(movie_id), str(movie_jpg)

    # step4-------------PMF
    # MF_SGD = ExplicitMF(train_data_matrix, 40, learning = 'sgd', verbose = True)

    # iter_array = [1, 2, 5, 10, 25]

    # MF_SGD.calculate_learning_curve(iter_array, test_data_matrix, learning_rate=0.01)

    # print(iter_array)
    # print(MF_SGD.test_mse)
    DNNPMF(train_data_matrix)