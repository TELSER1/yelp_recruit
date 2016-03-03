import json
import pickle
from keras.models import Sequential,Graph
from keras.layers.core import Flatten, Dense,Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import caffe
import pickle
import boto
def parse_googlenet(prototxt,caffemodel):
    net=caffe.Net('caffe/models/bvlc_googlenet/deploy.prototxt', 'caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel', caffe.TEST)
    layers_caffe = dict(zip(list(net._layer_names), net.layers))
    parameters={}
    parameters['conv1']={}
    parameters['conv1']['b']=layers_caffe['conv1/7x7_s2'].blobs[1].data
    parameters['conv1']['w']=layers_caffe['conv1/7x7_s2'].blobs[0].data
    parameters['conv2']={}
    parameters['conv2']['b']=layers_caffe['conv2/3x3_reduce'].blobs[1].data
    parameters['conv2']['w']=layers_caffe['conv2/3x3_reduce'].blobs[0].data
    parameters['conv3']={}
    parameters['conv3']['b']=layers_caffe['conv2/3x3'].blobs[1].data
    parameters['conv3']['w']=layers_caffe['conv2/3x3'].blobs[0].data
    parameters['conv4']={}
    parameters['conv4']['b']=layers_caffe['inception_3a/1x1'].blobs[1].data
    parameters['conv4']['w']=layers_caffe['inception_3a/1x1'].blobs[0].data
    parameters['conv5']={}
    parameters['conv5']['b']=layers_caffe['inception_3a/3x3_reduce'].blobs[1].data
    parameters['conv5']['w']=layers_caffe['inception_3a/3x3_reduce'].blobs[0].data
    parameters['conv6']={}
    parameters['conv6']['b']=layers_caffe['inception_3a/3x3'].blobs[1].data
    parameters['conv6']['w']=layers_caffe['inception_3a/3x3'].blobs[0].data
    parameters['conv7']={}
    parameters['conv7']['b']=layers_caffe['inception_3a/5x5_reduce'].blobs[1].data
    parameters['conv7']['w']=layers_caffe['inception_3a/5x5_reduce'].blobs[0].data
    parameters['conv8']={}
    parameters['conv8']['b']=layers_caffe['inception_3a/5x5'].blobs[1].data
    parameters['conv8']['w']=layers_caffe['inception_3a/5x5'].blobs[0].data
    parameters['conv9']={}
    parameters['conv9']['b']=layers_caffe['inception_3a/pool_proj'].blobs[1].data
    parameters['conv9']['w']=layers_caffe['inception_3a/pool_proj'].blobs[0].data
    return
def parse_alexnet(prototxt=None,caffemodel=None):
    net=caffe.Net('caffe/models/bvlc_alexnet/deploy.prototxt', 'caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel', caffe.TEST)
    layers_caffe = dict(zip(list(net._layer_names), net.layers))
    parameters={}
    parameters['conv1']={}
    parameters['conv1']['b']=layers_caffe['conv1'].blobs[1].data
    parameters['conv1']['w']=layers_caffe['conv1'].blobs[0].data
    parameters['conv2']={}
    parameters['conv2']['b']=layers_caffe['conv2'].blobs[1].data
    parameters['conv2']['w']=layers_caffe['conv2'].blobs[0].data
    parameters['conv3']={}
    parameters['conv3']['b']=layers_caffe['conv3'].blobs[1].data
    parameters['conv3']['w']=layers_caffe['conv3'].blobs[0].data
    parameters['conv4']={}
    parameters['conv4']['b']=layers_caffe['conv4'].blobs[1].data
    parameters['conv4']['w']=layers_caffe['conv4'].blobs[0].data
    parameters['conv5']={}
    parameters['conv5']['b']=layers_caffe['conv5'].blobs[1].data
    parameters['conv5']['w']=layers_caffe['conv5'].blobs[0].data
    parameters['fc1']={}
    parameters['fc1']['b']=layers_caffe['fc6'].blobs[1].data
    parameters['fc1']['w']=layers_caffe['fc6'].blobs[0].data    
    parameters['fc2']={}
    parameters['fc2']['b']=layers_caffe['fc7'].blobs[1].data
    parameters['fc2']['w']=layers_caffe['fc7'].blobs[0].data
    parameters['fc3']={}
    parameters['fc3']['b']=layers_caffe['fc8'].blobs[1].data
    parameters['fc3']['w']=layers_caffe['fc8'].blobs[0].data
    pickle.dump(parameters,open("alexnetparams.pkl",'wb'))
    s3=boto.connect_s3()
    bucket=s3.get_bucket("timpigpractice")
    key=bucket.new_key("alexnet")
    key.set_contents_from_filename("alexnetparams.pkl")
    return parameters
parse_alexnet()
