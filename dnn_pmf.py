# coding:utf-8

import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import time
import tensorflow as tf
import math

np.seterr(invalid='ignore')

weight_file = './vgg16_weights.npz'

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

def read_dataset():
    # step1----------read dataset
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('./v2_filter_user_ratedmovies.txt', sep = '\t', names = header)

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

def read_validId():
    valid_movieid = {}
    with open('./valid_movieid_imdbid.txt', 'rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            movieid, imdbid = temp[0], temp[1]
            valid_movieid[str(movieid)] = str(imdbid)
    return valid_movieid

# sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
# mse
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


from scipy.misc import imread, imresize
def save_all_picfeats(ratings,iindex_2_iid,valid_movieid):
    ferrlog = open('./error_log.txt', 'a+')
    n_users, n_items = ratings.shape
    base_frame_path = './movielens2011_frames/'
    base_feats_path = './movielens2011_cnnfeats/'
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None,224,224,3])
    vgg = vgg16(imgs, './vgg16_weights.npz', sess)
    for i in range(n_items):
        movieid = str(iindex_2_iid[i])
        imdbid = str(valid_movieid[movieid])
        print imdbid
        frame_path = base_frame_path + imdbid + '/'
        feats_path = base_feats_path + imdbid +'/'
        if not os.path.exists(feats_path):
            os.makedirs(feats_path)
        for root, dirs, files in os.walk(frame_path):
            if len(files) == 0:
                print '====empty===='
                continue
            else:
                for i,j in zip(range(len(files)), files):
                    jpg_path = frame_path + str(j)
                    # print jpg_path
                    s = str(j)
                    ss = s.replace('jpg','npy')
                    save_np_path = feats_path + str(ss)
                    print save_np_path
                    # get features
                    img = imread(jpg_path, mode='RGB')
                    img = imresize(img,(224,224))
                    feature=sess.run(vgg.fc3l, feed_dict ={vgg.imgs: [img]})
                    feature_vector=np.squeeze(feature)
                    feature_vector=feature_vector.reshape(1,1000)
                    np.save(save_np_path,feature_vector)
    ferrlog.close()

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

# get cnn_vecs by path
'''
def get_cnn_vecs(frame_path):
    ferrlog = open('./error_log.txt', 'a+')
    n_imgs = 0
    for root, dirs, files in os.walk(frame_path):
        n_imgs = len(files)
    # initialization
    cnn_vecs = np.zeros((n_imgs,1000))
    
    for root, dirs, files in os.walk(frame_path):
        if len(files) == 0:
            print '--------this folder is empty-------'
            ferrlog.write(str(frame_path)+'\n')
            continue
        else:
            for i,j in zip(range(n_imgs),files):
                jpg_path = frame_path + str(j)
                # print jpg_path
                print i
                each_cnn = get_picfeature(jpg_path)
                each_cnn = each_cnn.reshape(1,1000)
                cnn_vecs[i] = each_cnn
    ferrlog.close()
    return n_imgs, cnn_vecs
'''

# get one movie's imgs's cnn_vecs
def get_eachmovie_cnnfeats(base_feat_path):
    n_imgs = 0
    for root, dirs, files in os.walk(base_feat_path):
        n_imgs = len(files)
    # initialize the cnn_vecs
    cnn_vecs = np.zeros((n_imgs,1000))
    for root,dirs,files in os.walk(base_feat_path):
        for i,j in zip(range(n_imgs),files):
            feat_path = base_feat_path + str(j)
            # print feat_path
            each_cnn = np.load(feat_path)
            # print each_cnn.shape  #(1*1000)
            cnn_vecs[i,:] = each_cnn
    return n_imgs,cnn_vecs

# dnn+pmf
import os
def DNNPMF(ratings, iindex_2_iid, valid_movieid, n_factors=40, learning_rate=0.01, _lambda_1=0.01, _lambda_2=0.01, alpha=0.01):
    # ratings: ndarray
    # 1.define an indicator matrix I, I_ij = 1 if R_ij > 0 and 0 otherwise
    I_indicator = ratings
    I_indicator[I_indicator>0] = 1
    I_indicator[I_indicator<=0] = 0
    print I_indicator.shape

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
        size=(n_factors, n_users))
    item_vecs = np.random.normal(scale=1./n_factors, \
        size=(n_factors, n_items))
    # print user_vecs, item_vecs
    # (2113, 40) (8887, 40)
    print user_vecs.shape, item_vecs.shape


    # 5.partial train sgd------during this, [get CNN vec of images 1*1000]
    t = time.time()
    base_path = './movielens2011_cnnfeats/'
    
    q_vecs=np.random.normal(scale=1./n_factors, size=(n_factors, 1000))

    q_present=np.zeros((n_factors,1000))
    # # Update Q & B
    # for i in range(n_items):
    #     # 正例
    #     movieid = str(iindex_2_iid[i])
    #     imdbid = str(valid_movieid[movieid])
    #     # 反例
    #     if i != (n_items-1): 
    #         movieid_neg=str(iindex_2_iid[i+1])
    #         imdbid_neg=str(valid_movieid[movieid_neg])
    #     else:
    #         movieid_neg=str(iindex_2_iid[i-1])
    #         imdbid_neg=str(valid_movieid[movieid_neg])
    #     print imdbid

    #     base_feat_path = base_path + imdbid + '/'
    #     base_feat_path_neg = base_path + imdbid_neg + '/'

    #     # according to the path, getting the cnn_vecs of each movie
    #     n_imgs, cnn_vecs = get_eachmovie_cnnfeats(base_feat_path)
    #     n_imgs_neg, cnn_vecs_neg = get_eachmovie_cnnfeats(base_feat_path_neg)

    #     q_present_eachimg=np.zeros((n_factors,1000))
    #     for j in range(n_imgs):
    #         # 每张p_{v_j}^s都randomly产生r张negative sample作为反例
    #         # 选下一个电影的关键帧作为negative sample

    #         # print item_vecs[:,i].shape
    #         # print cnn_vecs[j,:].shape
    #         qa=np.array([item_vecs[:,i]])
    #         qb=np.array([cnn_vecs[j,:]])
    #         # print qa.shape
    #         # print qb.shape
    #         qx = 1-sigmoid((qa.dot(q_vecs)).dot(qb.T))
    #         # x = 1-sigmoid((((item_vecs[:,i]).T).dot(q_vecs)).dot(cnn_vecs[j,:]))

    #         q_present_one = qx*((qa.T).dot(qb))
    #         # present_one=x*((item_vecs[:,i]).dot((cnn_vecs[j,:]).T))
    #         # print present_one.shape
    #         # type(qx): float
    #         # type(q_present_one): ndarray

    #         # qy = float(0)
    #         q_present_two = np.zeros((n_factors,1000))
    #         for k in range(n_imgs_neg):
    #             qc = np.array([cnn_vecs_neg[k,:]])
    #             qy = 1-sigmoid(-1*qa.dot(q_vecs).dot(qc.T))
    #             # y += 1-sigmoid((-1*((item_vecs[:,i]).T).dot(q_vecs)).dot(cnn_vecs_neg[k]))

    #             q_present_two += qy*((qa.T).dot(qc)) 
    #         # present_two = y*item_vecs[:,i].dot((cnn_vecs_neg[k]).T)
    #         # print present_two.shape
    #         # 得到一张图片和它的反例的特征表达式
    #         q_present_eachimg += q_present_one - q_present_two
    #         # 叠加得出一部电影的所有图片和反例的特征表达式
    #     # 得到所有电影和它的反例的特征表达式
    #     q_present += q_present_eachimg

    # q_vecs += learning_rate*(alpha * q_present - 2 * _lambda_2 * q_vecs)
    # print 'Training Q elapsed:\t',time.time()-t

    # Update B
    t = time.time()
    b_vecs = np.zeros((n_factors,n_items))
    print b_vecs.shape #(40*m)
    for i in range(n_items):
        # 正例
        movieid = str(iindex_2_iid[i])
        imdbid = str(valid_movieid[movieid])

        # 反例
        if i != (n_items-1):
            movieid_neg = str(iindex_2_iid[i+1])
            imdbid_neg = str(valid_movieid[movieid_neg])
        else:
            movieid_neg = str(iindex_2_iid[i-1])
            imdbid_neg = str(valid_movieid[movieid_neg])
        print imdbid

        base_feat_path = base_path + imdbid + '/'
        base_feat_path_neg = base_path + imdbid_neg + '/'

        n_imgs, cnn_vecs = get_eachmovie_cnnfeats(base_feat_path)
        n_imgs_neg, cnn_vecs_neg = get_eachmovie_cnnfeats(base_feat_path_neg)

        b_present = np.zeros((n_factors,1))
        for j in range(n_imgs):
            ba = np.array([item_vecs[:,i]])
            bb = np.array([cnn_vecs[j,:]])
            # print ba.shape
            # print bb.shape
            bx = 1-sigmoid((ba.dot(q_vecs)).dot(bb.T))

            b_present_one = bx*(q_vecs.dot(bb.T))

            by = float(0)
            b_present_two = np.zeros((n_factors,1))
            for k in range(n_imgs_neg):
                bc = np.array([cnn_vecs_neg[k,:]])
                by += 1-sigmoid(-1*ba.dot(q_vecs).dot(bc.T))

                b_present_two += by*(q_vecs.dot(bc.T))

            b_present += b_present_one - b_present_two
            # print b_present.shape  #(40*1)
        # 由于ndarray矩阵不能直接列赋值 所以用其转置矩阵进行行赋值
        # print b_present.shape  #(40*1)
        # print b_vecs.shape #(40*m)
        b_vecs_T = b_vecs.T
        # print b_vecs_T.shape   #(m*40)
        # print b_present.T.shape  #(1*40)
        b_vecs_T[i] = b_present.T
    b_vecs = b_vecs_T.T

        



    
    print 'Training B elapsed:\t',time.time()-t
    
    n_iter = 30
    ctr = 1
    while ctr <= n_iter:
        if ctr % 10 ==0:
            print 'current iteration:{}'.format(ctr)
        # training_indices = np.arange(n_samples)
        # np.random.shuffle(training_indices)
        # sgd(ratings, sample_row, sample_col, training_indices, user_vecs, item_vecs)
        # sgd
        # Update U
        user_vecs+=learning_rate*(2*item_vecs.dot((I_indicator*ratings).T) \
            -2*item_vecs.dot((I_indicator*((user_vecs.T).dot(item_vecs))).T) \
            -2*_lambda_1*user_vecs)


        #  Update V
        item_vecs+=learning_rate*(2*user_vecs.dot(I_indicator*ratings) \
            -2*user_vecs.dot(I_indicator*((user_vecs.T).dot(item_vecs))) \
            -2*_lambda_1*item_vecs+alpha*b_vecs)

        ctr += 1

# sgd predict
def predict(user_vecs, item_vecs, u, i):
    return user_vecs[:, u].dot(item_vecs[:, i].T)

# predict ratings for every user and item
def predict_all(user_vecs, item_vecs):
    predictions = np.zeros((user_vecs.shape[0], item_vecs.shape[0]))
    for u in range(user_vecs.shape[0]):
        for i in range(item_vecs.shape[0]):
            predictions[u, i] = predict(user_vecs, item_vecs, u, i)

    return predictions

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
    # print type(train_data_matrix)

    # step4---------calculate sparsity
    print calc_sparsity(df, n_users, n_items)
    print 'step4--calculate sparsity\telapse:', time.time() - t
    
    # step5----------read valid_imdbid
    valid_movieid = read_validId()

    # reverse dictionary
    iindex_2_iid = dict((value,key) for key,value in iid_2_iindex.iteritems())

    # step6-----------save feats
    # save_all_picfeats(train_data_matrix, iindex_2_iid, valid_movieid)
   
    # path='./1.jpg' #test
    # cnn=get_picfeature(path)#test
    # print type(cnn)#test
    # print cnn.shape#test
    # print str(iindex_2_iid[3]) #test
    # print valid_movieid[str(iindex_2_iid[3])] #test
    # print iindex_2_iid[7257] #test
    # print valid_movieid #test
    # print train_data_matrix #test
    # print test_data_matrix #test

    # step6----------dnn_pmf

    DNNPMF(train_data_matrix, iindex_2_iid, valid_movieid)  