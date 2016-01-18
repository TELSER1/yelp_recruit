import pandas as pd
import numpy as np
import os
import pdb
from scipy.misc import imread
from sklearn.metrics import log_loss
from sklearn.preprocessing import MultiLabelBinarizer
from seya.layers.attention import SpatialTransformer
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import f1_score
import pdb
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))
def extract_images(x):
    x['random']=np.random.uniform(size=(x.shape[0],1))
    return(x.sort('random').head(10)['photo_id'].tolist())
def loadbusimage(x):
    bisimg=[np.array(imread("train_photos/train244/"+str(f_)+".jpg")).astype(np.float32) for f_ in x]
    for i in xrange(0,10-len(bisimg)):
        bisimg.append(np.zeros(shape=(224,224,3)).astype(np.float32))
    return(np.concatenate(bisimg,axis=2))
def loadimages(x):
    a_=[loadbusimage(im_) for im_ in x]
    return(np.array(a_))
labels=pd.read_csv("train.csv")
labels=labels[pd.isnull(labels['labels'])==False]
bismatch=pd.read_csv("train_photo_to_biz_ids.csv")
photo_labels=bismatch.merge(labels,how='left',on='business_id')
photo_labels=photo_labels[pd.isnull(photo_labels['labels'])==False]
photo_labels['labels']=photo_labels['labels'].map(lambda x:[int(i) for i in x.split(" ")])
np.random.seed(42)
labels['assignment']=np.random.randint(0,10,size=(labels.shape[0],1))
photo_labels=photo_labels.merge(labels[['business_id','assignment']],on='business_id')
train=photo_labels[photo_labels['assignment']<=7].reset_index(drop=True)
test=photo_labels[photo_labels['assignment']>7].reset_index(drop=True)

mlb=MultiLabelBinarizer()
mlb.fit(train['labels'].tolist()+test['labels'].tolist())
#INSERT NORMALIZATION TRAINING HERE
nfilters=32
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]
locnet = Sequential()
locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=(224, 224,30)))
locnet.add(Convolution2D(20, 5, 5))
locnet.add(MaxPooling2D(pool_size=(2,2)))
locnet.add(Convolution2D(20, 5, 5))

locnet.add(Flatten())
locnet.add(Dense(50))
locnet.add(Activation('relu'))
locnet.add(Dense(6, weights=weights))
model=Sequential()
model.add(SpatialTransformer(localization_net=locnet,input_shape=(224, 224,30)))
model.add(Convolution2D(nfilters, 3, 3,border_mode='valid',dim_ordering='tf'))
model.add(Activation('relu'))
model.add(Convolution2D(nfilters, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(nfilters, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(nfilters, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(.5))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Dense(9))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.001, decay=1e-8, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#split into train/test sets, load in ten images per business
epochs=4000
for epoch in xrange(0,epochs):
    tphotos=train.groupby('business_id').apply(extract_images).reset_index()
    tphotos.reindex(np.random.permutation(tphotos.index))
    tphotos.columns=['business_id','photo_id']
    tphotos=tphotos.merge(labels,on='business_id',how='left')
    tphotos['labels']=tphotos['labels'].map(lambda x:[int(i) for i in x.split(" ")])
    tstphotos=test.groupby('business_id').apply(extract_images).reset_index()
    tstphotos.reindex(np.random.permutation(tstphotos.index))
    tstphotos.columns=['business_id','photo_id']
    tstphotos=tstphotos.merge(labels,on='business_id',how='left')
    tstphotos['labels']=tstphotos['labels'].map(lambda x:[int(i) for i in x.split(" ")])
    #Y_train=mlb.transform(tphotos['labels'])
    if epoch==0:
        Y_train=loadimages(tstphotos['photo_id'])
        Y_test=mlb.transform(tstphotos['labels'])
    X_test=mlb.transform(tphotos['labels'])
    X_train=loadimages(tphotos['photo_id'])
    model.fit(X_train,X_test,batch_size=128,nb_epoch=1,verbose=0)
    pred=model.predict(Y_train)
    probs=model.predict_proba(Y_train)
    print probs.mean(axis=0)
    print probs.max(axis=0)
    print probs.min(axis=0)
    print f1_score(Y_test,pred)
    print epoch
