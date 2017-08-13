# coding:utf-8
import pandas as pd
import numpy as np
import time
import math

data_dir = './hetrec2011-lastfm-2k/'

# Parameters
context_weight
lamda
gamma
dis_coef
alpha

def read_dataset():
    # step1----------read dataset
    header = ['userID', 'artistID', 'weight']
    df = pd.read_csv(data_dir + 'user_artists.dat', sep = '\t', names = header)

    users = df.userID.unique()
    items = df.artistID.unique()

    n_users = users.shape[0]
    n_items = items.shape[0]

    print type(users)
    print users
    print items
    print df.head(3)
    print df.describe()
    print n_users, n_items
    return df, users, items, n_users, n_items

def read_context():
    header = ['userID', 'artistID']
    df = pd.read_csv(data_dir + 'user_friends.dat', seq = '\t', names = header)

def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

def embedding_learning(train_data):
    # User-Context
    UC = np.random.normal(0.0, 0.01, (n_users, K))
    # Item-Context
    IC = np.random.normal(0.0, 0.01, (n_items, K))

    log_likelihood = 0.0

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

                IC[i] += gamma * ((1 - sigmoid(z)) * 2 * alpha * wc * (UC[u_i] - UC[u_j]) - 2 * lamda * IC[i])
                UC[u_i] += gamma * ((1 - sigmoid(z)) * 2 * alpha * wc * (IC[i] - UC[u_i]) - 2 * lamda * UC[u_i])
                UC[u_j] += gamma * ((1 - sigmoid(z)) * 2 * alpha * wc * (IC[i] - UC[u_j]) - 2 * lamda * UC[u_j])
        print 'Iter: %d    likelihood: %f   elapsed:  %fseconds'%(iteration, log_likelihood, time.time() - t)
