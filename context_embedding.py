# coding:utf-8
import pandas as pd
import numpy as np
import time
import math

data_dir = './hetrec2011-lastfm-2k/'
model_dir = './context_model/'

# Parameters
lamda = 0.03
learning_rate = 0.005
dis_coef = 0.25
alpha = 0.2
max_iters = 50

def read_dataset():
    train_data = []
    user_set = set()
    user_list = []
    artist_set = set()
    artist_list = []

    with open(data_dir + 'user_artists_edit.dat', 'rb') as fp:
        lines = fp.readlines()
        for line  in lines:
            tempdata = line.strip().split('\t')
            userid, artistid, weight = int(tempdata[0]), int(tempdata[1]), int(tempdata[2])
            train_data.append([userid, artistid, weight])
            user_set.add(userid)
            artist_set.add(artistid)
        
        user_list = list(user_set)
        user_list.sort()
        print user_list
        artist_list = list(artist_set)
        artist_list.sort()
        print artist_list

    return train_data, user_list, artist_list

# read_dataset()

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

def embedding_learning(train_data):
    # User-Context
    UC = np.random.normal(0.0, 0.01, (n_users, K))
    # Item-Context
    IC = np.random.normal(0.0, 0.01, (n_items, K))

    log_likelihood = 0.0

    try:
        for iteration in range(max_iters):
            t = time.time()
            for each_data in train_data:
                u_i, i, w_i = each_data
                for u_j in context_list[u_i]:
                    Di = np.linalg.norm(IC[i] - UC[u_i]) ** 2
                    Dj = np.linalg.norm(IC[i] - UC[u_j]) ** 2

                    z = Dj - Di

                    wc = w_i ** dis_coef
                    
                    log_likelihood += np.log(sigmoid(z))

                    IC[i] += learning_rate * ((1 - sigmoid(z)) * 2 * alpha * wc * (UC[u_i] - UC[u_j]) - 2 * lamda * IC[i])
                    UC[u_i] += learning_rate * ((1 - sigmoid(z)) * 2 * alpha * wc * (IC[i] - UC[u_i]) - 2 * lamda * UC[u_i])
                    UC[u_j] += learning_rate * ((1 - sigmoid(z)) * 2 * alpha * wc * (IC[i] - UC[u_j]) - 2 * lamda * UC[u_j])
            print 'Iter: %d    likelihood: %f   elapsed:  %fseconds'%(iteration, log_likelihood, time.time() - t)
    finally:
        np.save(model_dir + 'IC', IC)
        np.save(model_dir + 'UC', UC)
        print 'Model saved...'