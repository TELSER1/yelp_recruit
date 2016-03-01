from keras.models import Sequential,Graph
from keras.layers.core import Flatten, Dense,Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import pdb
import h5py
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def VGG_16_shared(n_photos=1):
    model = Graph()
    model2 = VGG_16('vgg16_weights.h5')
    for i in xrange(0,n_photos):
        model.add_node(ZeroPadding2D((1,1),input_shape=(3,224,224)),name='input'+str(i))
    model.add_shared_node(Convolution2D(64, 3, 3, activation='relu',weights=model2.layers[1].get_weights()),name='conv1',inputs=['input'+str(i) for i in xrange(0,n_photos)])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero1',inputs=['conv1'])
    model.add_shared_node(Convolution2D(64, 3, 3, activation='relu',weights=model2.layers[3].get_weights()),name='conv2',inputs=['zero1'])
    model.add_shared_node(MaxPooling2D((2,2), strides=(2,2)),name='max1',inputs=['conv2'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero2',inputs=['max1'])
    model.add_shared_node(Convolution2D(128, 3, 3, activation='relu',weights=model2.layers[6].get_weights()),name='conv3',inputs=['zero2'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero3',inputs=['conv3'])
    model.add_shared_node(Convolution2D(128, 3, 3, activation='relu',weights=model2.layers[8].get_weights()),name='conv4',inputs=['zero3'])
    model.add_shared_node(MaxPooling2D((2,2), strides=(2,2)),name='max2',inputs=['conv4'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero4',inputs=['max2'])
    model.add_shared_node(Convolution2D(256, 3, 3, activation='relu',weights=model2.layers[11].get_weights()),name='conv5',inputs=['zero4'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero5',inputs=['conv5'])
    model.add_shared_node(Convolution2D(256, 3, 3, activation='relu',weights=model2.layers[13].get_weights()),name='conv6',inputs=['zero5'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero6',inputs=['conv6'])
    model.add_shared_node(Convolution2D(256, 3, 3, activation='relu',weights=model2.layers[15].get_weights()),name='conv7',inputs=['zero6'])
    model.add_shared_node(MaxPooling2D((2,2), strides=(2,2)),name='max3',inputs=['conv7'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero7',inputs=['max3'])
    model.add_shared_node(Convolution2D(512, 3, 3, activation='relu',weights=model2.layers[18].get_weights()),name='conv8',inputs=['max3'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero8',inputs=['conv8'])
    model.add_shared_node(Convolution2D(512, 3, 3, activation='relu',weights=model2.layers[20].get_weights()),name='conv9',inputs=['zero8'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero9',inputs=['conv9'])
    model.add_shared_node(Convolution2D(512, 3, 3, activation='relu',weights=model2.layers[22].get_weights()),name='conv10',inputs=['zero9'])
    model.add_shared_node(MaxPooling2D((2,2), strides=(2,2)),name='max4',inputs=['conv10'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero10',inputs=['conv10'])
    model.add_shared_node(Convolution2D(512, 3, 3, activation='relu',weights=model2.layers[25].get_weights()),name='conv11',inputs=['zero10'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero11',inputs=['conv11'])
    model.add_shared_node(Convolution2D(512, 3, 3, activation='relu',weights=model2.layers[27].get_weights()),name='conv12',inputs=['zero11'])
    model.add_shared_node(ZeroPadding2D((1,1)),name='zero12',inputs=['conv12'])
    model.add_shared_node(Convolution2D(512, 3, 3, activation='relu',weights=model2.layers[29].get_weights()),name='conv13',inputs=['zero12'])
    model.add_shared_node(MaxPooling2D((2,2), strides=(2,2)),name='max5',inputs=['conv13'])
    model.add_shared_node(Flatten(),name='flatten',inputs=['max5'])
    model.add_shared_node(Dense(4096, activation='relu'),name='dense1',inputs=['flatten'])
    model.add_shared_node(Dropout(0.5),name='dropout1',inputs=['dense1'])
    model.add_shared_node(Dense(4096, activation='relu'),name='dense2',inputs=['dropout1'])
    model.add_shared_node(Dropout(0.5),name='dropout2',inputs=['dense2'])
    model.add_shared_node(Dense(1000, activation='softmax'),name='output',inputs=['dropout2'])
    return model
def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()
#mod=VGG_16_shared(n_photos=1)
#print_structure('vgg16_weights.h5')
#model = VGG_16('vgg16_weights.h5')
#model2=VGG_16()
#pdb.set_trace()
