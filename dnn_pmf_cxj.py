# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from numpy.linalg import solve
import tensorflow as tf
import scipy.io as sio
from scipy.misc import imread, imresize
import numpy as np
import time
import math

weight_file = './vgg16_weights.npz'

# sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
# vgg16 model
class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

# mse
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

# MF function
class ExplicitMF:
    def __init__(self, ratings, cnn, n_factors=40, learning='sgd', _lambda_1=0.01, _lambda_2=0.01, \
     alpha=0.01, item_fact_reg=0.01, user_fact_reg=0.01,
                  verbose=False):
    """    
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        ratings matrix which is ~ user x item

        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model
        learning : (str)
            Method of optimization. Options include
            'sgd' or 'als'.

        item_fact_reg : (float)
            Regularization term for item latent factors

        user_fact_reg : (float)
            Regularization term for user latent factors

        item_bias_reg : (float)
            Regularization term for item biases

        user_bias_reg : (float)
            Regularization term for user biases

        verbose : (bool)
            Whether or not to printout training progress
    """    

        self.ratings = ratings # (ndarray)  user x item matrix with corresponding ratings
        self.cnn = cnn
        self.n_users, self.n_items = ratings.shape # Number of users and items
        self.n_factors = n_factors # Number of latent factors to use in MF model
        self.item_fact_reg = item_fact_reg # Regularization term for item latent factors
        self.user_fact_reg = user_fact_reg # Regularization term for user latent factors
        # self.mean_avg = np.mean(ratings)
        # define an indicator matrix Y, Y_ij = 1 if R_ij > 0 and 0 otherwise
        # self.y_indicator = ratings
        # self.y_indicator[y_indicator>0] = 1
        self._lambda_1 = _lambda_1
        self._lambda_2 = _lambda_2
        self.alpha = alpha

        self.item_bias_reg = item_bias_reg # Regularization term for item biases
        self.user_bias_reg = user_bias_reg # Regularization term for user biases
        self.learning = learning
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose
'''
    def als_step(self,
                 latent_vectors,
                 ratings,
                 fixed_vecs,
                 _lambda,
                 type='user'):
        # One of the two ALS steps. Solve for the latent vectors specified by type.
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda

            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI),
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors
'''
    def train(self, n_iter = 10, learning_rate = 0.1):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        # U: K*n  V: K*m
        self.user_vecs = np.random.normal(scale=1. / self.n_factors,  \
            size=(self.n_factors, self.n_users))
        self.item_vecs = np.random.normal(scale=1. / self.n_factors,  \
            size=(self.n_factors, self.n_items))
        # V*P*CNN(p)   CNN: 1*1000
        # eg: V^k*1 = V^k*m * P^m*1000 * (CNN^T)^1000*1
        self.pic_vecs = np.random.rand(self.n_factors, 1000)
        self.b_vecs = np.random.rand(self.n_factors, self.n_items)
        # q: k*d the interaction matrix between visual contents and latent movie features
        self.q_vecs = np.random.rand(self.n_factors, 1000)


        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            # Update P k*1000  k = n_factors
            # self.pic_bias = np.zeros(self.n_factors)

            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
            self.partial_train(n_iter)

    def partial_train(self, n_iter):
        # Train model for n_iter iterations. Can be called multiple times for further training.
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print('\tcurrent iteration: {}'.format(ctr))
            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs,
                                               self.item_vecs,
                                               self.ratings,
                                               self.user_fact_reg,
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs,
                                               self.user_vecs,
                                               self.ratings,
                                               self.item_fact_reg,
                                               type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u, i] - prediction)  # error
            # define an indicator matrix Y, Y_ij = 1 if R_ij > 0 and 0 otherwise
            # y_indicator: n*m
            y_indicator = self.ratings
            y_indicator[y_indicator>0] = 1 
            # # Update biases
            self.user_bias[u] += self.learning_rate * (e - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * (e - self.item_bias_reg * self.item_bias[i])

            # Update latent factors
            # self.user_vecs[u, :] += self.learning_rate * (e * self.item_vecs[i, :] - self.user_fact_reg * self.user_vecs[u, :])
            self.user_vecs[:, u] += self.learning_rate * (2 * self.item_vecs[:, i].dot(y_indicator * self.ratings).T) \
             - 2 * (self.item_vecs[:, i].dot(y_indicator * (self.user_vecs[:, u].T * self.item_vecs[:, i])).T) \
             - 2 * self._lambda_1 * self.user_vecs[:, u]
            # self.item_vecs[i, :] += self.learning_rate * (e * self.user_vecs[u, :] - self.item_fact_reg * self.item_vecs[i, :])
            self.item_vecs[:, i] += self.learning_rate * (2 * self.user_vecs[:, u].dot(y_indicator * self.ratings).T) \
             - 2 * (self.user_vecs[:, u]).dot(y_indicator * (self.user_vecs[:, u].T * self.item_vecs[:, i])) \
             - 2 * self._lambda_1 * self.item_vecs[:, i] + self. alpha * b_vecs[:, i] 
            # Update b_vecs: k*m 先使用随机初始化来测验 最后调成negative sampling 
            self.b_vecs = np.random.rand(self.n_factors, self.n_items)
            # Update Q interaction matrix between V and CNN 每读到一个item（v）就更新
            # 每一次更新的UV都伴随着更新对应的一个Q
            for n in range(self.n_factors):
                self.q_vecs[n, :] += self.alpha * self.item_vecs[:, i] * (1 - sigmoid(self.user_vecs[:, u].T))
            

            # self.pic_vecs[, :]

    def predict(self, u, i):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            # avg_rate = np.mean(self.ratings[:, u])

            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            # 加上一个视觉特征
            # self.item_vecs[i, :] = pic_latvecs[]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            return prediction

    def predict_all(self):
        """ Predict ratings for every user and item."""
        predictions = np.zeros((self.user_vecs.shape[0],
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)

        return predictions

    def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
        """
        Keep track of MSE as a function of training iterations.

        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).

        The function creates two new class attributes:

        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse = []
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [get_mse(predictions, self.ratings)]
            self.test_mse += [get_mse(predictions, test)]
            if self._v:
                print('Train mse: ' + str(self.train_mse[-1]))
                print('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter



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

def get_picfeature(poster_path):
    # step5------------use vgg16
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, './vgg16_weights.npz', sess)

    img = imread(poster_path, mode = 'RGB')
    img = imresize(img, (224, 224))
    feature = sess.run(vgg.fc3l, feed_dict ={vgg.imgs: [img]})
    # feature = np.reshape(feature, [7, 7, 512])
    # dic = {'features': feature}
    # sio.savemat('./features/' + str(movie_id) + '.mat', dic)
    # 将结果压缩成一个一维数组的特征向量其实不需要 全连接层最后得到的就是一个一维向量
    feature_vector = np.squeeze(feature)
    # 1*1000 vector 1dim
    # print feature_vector.shape
    # print feature_vector.ndim
    # print feature_vector
    return feature_vector

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

    # step4---------calculate sparsity
    print calc_sparsity(df, n_users, n_items)
    print 'step4--calculate sparsity\telapse:', time.time() - t

    # test5
    poster_path = './1.jpg'
    feature_vec = get_picfeature(poster_path)
    print '-----'
    


    # step5---------use vgg16 to extract the pic features
    # with open('./movie_ID_Jpg.txt', 'rb') as fopen:
    #     lines = fopen.readlines()
    #     for line in lines:
    #         tempdata = line.strip().split('\t')
    #         movie_id, movie_jpg = tempdata[0], tempdata[1]
            
    #         print str(movie_id), str(movie_jpg)

    # step4-------------PMF
    # MF_SGD = ExplicitMF(train_data_matrix, 40, learning = 'sgd', verbose = True)

    # iter_array = [1, 2, 5, 10, 25, 50, 100]

    # MF_SGD.calculate_learning_curve(iter_array, test_data_matrix, learning_rate=0.01)

    # print(iter_array)
    # print(MF_SGD.test_mse)