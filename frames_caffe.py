# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
# import caffe
# import sys
import pickle
import struct
import sys,cv2

caffe_root = '/home/shouldchan/caffe/'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe

model_root = '/home/shouldchan/caffe/models/'

# 运行模型的prototxt
deployPrototxt =  '/home/shouldchan/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
# 相应载入的modelfile
modelFile = '/home/shouldchan/caffe/models/bvlc_reference_caffenet/caffe_rcnn_imagenet_model.caffemodel'
# meanfile 也可以用自己生成的
meanFile = '/home/shouldchan/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
# 需要提取的id列表
IMAGE_LIST_DIR = '/home/shouldchan/文档/CNN_caffe/Movielens/movielens2011/valid_movieid_imdbid.txt'

FOLDER_DIR = '/home/shouldchan/文档/CNN_caffe/Movielens/movielens2011/movielens2011_frames/'

VECTOR_DIR = '/home/shouldchan/文档/CNN_caffe/Movielens/'

# SIM_DIR = '/home/shouldchan/文档/CNN_caffe/Toro/CNN_Toro_similarity.txt'

postfix = '.csv'

# 初始化函数的相关操作
def initilize():
    print 'initilize ... '

    sys.path.insert(0, caffe_root + 'python')
    
    caffe.set_mode_cpu()

    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net  

# 读取文件列表
def read_path(IMAGE_LIST_DIR):
    folder_list = []

    with open(IMAGE_LIST_DIR,'r') as fi:
        lines = fi.readlines()
        for line in lines:
            tempData = line.strip().split('\t')
            movie_id, imdb_id = tempData[0], tempData[1]
            folderpath = FOLDER_DIR + str(imdb_id) + '/'
            folder_list.append(folderpath)
    print 'read folder_list done image num ', len(folder_list)
    print folder_list
    return folder_list

def read_extract_image(folder_list, net):
	for eachpath in folder_list:
		print eachpath
		for root, dirs, files in os.walk(eachpath):
			# print files
			extractFeature(eachpath, files, net)


		
# 提取特征并保存为相应地文件
def extractFeature(folderpath, image_list, net):
    # save each pic's features in each line of lists
    featAll = []
    # 对输入数据做相应地调整如通道、尺寸等等
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(meanFile).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))  
    # set net to batch size of 1 如果图片较多就设置合适的batchsize 
    net.blobs['data'].reshape(1,3,227,227)      #这里根据需要设定，如果网络中不一致，需要调整
    # num=0
    # print folderpath
    for imagefile in image_list:
        imagefile_abs = os.path.join(folderpath, imagefile)
        # imageId = imagefile.replace()
        # print imagefile_abs
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile_abs))
        out = net.forward()
        fea_file = imagefile_abs.replace('.jpg',postfix)
        # read the features of fc6
        feature = net.blobs['fc6'].data[0]
        # normalization
        feature_standard = (feature - min(feature))/(max(feature) - min(feature))
        # transfer to list
        tmpf = feature_standard.reshape(1,feature_standard.size)
        s = tmpf.tolist()
        feat=reduce(lambda x,y: x+y,s)
        # num += 1
        featAll.append([imagefile,feat])
        print imagefile
        # with open(fea_file,'wb') as f:
        #     for x in xrange(0, net.blobs['fc6'].data.shape[0]):
        #         for y in xrange(0, net.blobs['fc6'].data.shape[1]):
        #             f.write(struct.pack('f', net.blobs['fc6'].data[x,y]))
    # print featAll

    fwrite=open(folderpath + 'vector_frames.txt','a+')
    for [imagefile, line] in featAll:
        fwrite.write(str(imagefile)+'\t'+str(line)+'\n')

if __name__ == "__main__":
    net = initilize()
    folder_list = read_path(IMAGE_LIST_DIR) 
    read_extract_image(folder_list, net)
