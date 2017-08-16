# coding:utf-8
import pandas as pd
import numpy as np
import time
import math
import random
from keras.layers import Embedding
from keras import initializations
from keras.regularizers import l1, l2, l1l2
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten   
from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop 


data_dir = './hetrec2011-lastfm-2k/'
model_dir = './context_model/'

# Parameters
lamda = 0.03
learning_rate = 0.005
dis_coef = 0.25
alpha = 0.4
max_iters = 100
dim_num = 60


def read_dataset():
    all_data = []
    user_set = set()
    user_list = []
    artist_set = set()
    artist_list = []

    with open(data_dir + 'user_artists_edit.dat', 'rb') as fp:
        lines = fp.readlines()
        for line  in lines:
            tempdata = line.strip().split('\t')
            userid, artistid, weight = int(tempdata[0]), int(tempdata[1]), float(tempdata[2])
            weight = weight / 5000
            all_data.append([userid, artistid, weight])
            user_set.add(userid)
            artist_set.add(artistid)
        
        user_list = list(user_set)
        user_list.sort()
        n_users = len(user_set)
        # print user_list
        artist_list = list(artist_set)
        artist_list.sort()
        n_items = len(artist_set)
        # print artist_list

    user_dic = {}
    count_user = 0
    for i in user_list:
        user_dic[i] = count_user
        count_user += 1
    # print user_dic 

    artist_dic = {}
    count_artist = 0
    for i in artist_list:
        artist_dic[i] = count_artist
        count_artist += 1
    # print artist_dic
    return all_data, user_dic, artist_dic, n_users, n_items

# all_data, user_dic, artist_dic, n_users, n_items = read_dataset()

def make_train_test(all_data):
    test_data = []
    test_num = int(len(all_data) * 0.25)
    test_data = random.sample(all_data, test_num)
    print test_data
    train_data = [i for i in all_data if i not in test_data]
    n_train = len(train_data)
    n_test = len(test_data)
    print n_train, n_test
    return train_data, test_data, n_train, n_test

# make_train_test(all_data)

def read_context():
    context_dic = {}
    with open(data_dir + 'user_friends_edit.dat', 'rb') as fp:
        lines = fp.readlines()
        for line in lines:
            tempdata = line.strip().split('\t')
            userid, friendid = int(tempdata[0]), int(tempdata[1])
            context_dic.setdefault(userid, []).append(friendid)
        # print context_dic
    return context_dic

# read_context()


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

def embedding_learning(train_data, user_dic, artist_dic, context_list, n_users, n_items):
    # User embeddings
    UC = np.random.normal(0.0, 0.01, (n_users, dim_num))
    # Item embeddings
    IC = np.random.normal(0.0, 0.01, (n_items, dim_num))

    try:
        for iteration in range(max_iters):
            print 'loading...iteration: %d'%iteration
            t = time.time()

            for each_data in train_data:
                u_i, i, w_i = each_data
                w_i = w_i ** dis_coef
                # print artist_dic[i]
                for u_j in context_list[u_i]:

                    IC[artist_dic[i]] += learning_rate * ((1 - sigmoid(w_i)) * 2 * alpha  * (UC[user_dic[u_i]] - UC[user_dic[u_j]]) - 2 * lamda * IC[artist_dic[i]])
                    UC[user_dic[u_i]] += learning_rate * ((1 - sigmoid(w_i)) * 2 * alpha  * (IC[artist_dic[i]] - UC[user_dic[u_i]]) - 2 * lamda * UC[user_dic[u_i]])
                    UC[user_dic[u_j]] += learning_rate * ((1 - sigmoid(w_i)) * 2 * alpha  * (IC[artist_dic[i]] - UC[user_dic[u_j]]) - 2 * lamda * UC[user_dic[u_j]])
                    
                    # print IC[artist_dic[i]]
            print 'Iter: %d   elapsed:  %fseconds'%(iteration, time.time() - t)
    finally:
        np.save(model_dir + 'Item_Emb', IC)
        np.save(model_dir + 'User_Emb', UC)
        np.savetxt(model_dir + 'Item_Emb.txt', IC)
        np.savetxt(model_dir + 'User_Emb.txt', UC)
        print 'Model saved...'
def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

if __name__ == '__main__':
    all_data, user_dic, artist_dic, n_users, n_items = read_dataset()
    context_dic = read_context()
    # train_data, test_data, n_train, n_test = make_train_test(all_data)
    # embedding_learning(train_data, user_dic, artist_dic, context_dic, n_users, n_items)

    User_Embedding = Embedding(input_dim=n_users, output_dim=60, \
        name='user_embedding', init=init_normal,W_regularizer=l2(0),  \
        input_length=1)
    # print type(User_Embedding)

    Item_Embedding = Embedding(input_dim=n_items, output_dim=60, \
        name='item_embedding', init=init_normal,W_regularizer=l2(0), \
        input_length=1)

    model = Sequential()
    model.add(Embedding(n_items, 60, input_length=10))
    # model.add(Dense(32, input_dim=n_items))
    # model.add(Activation('relu'))

    input_array = np.random.randint(50000, size=(n_items, 10))
    print 'input_array:\t'
    print input_array.shape

    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile('rmsprop', 'mse')

    output_array = model.predict(input_array)
    print 'output_array:\t'
    print output_array.shape

    # x = all_data
    # y = 
    # fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,  \
    #     validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
