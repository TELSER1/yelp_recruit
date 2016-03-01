import pandas as pd
import numpy as np
import os
import pdb
from scipy.misc import imread,imresize
from sklearn.metrics import log_loss
from sklearn.preprocessing import MultiLabelBinarizer
from seya.layers.attention import SpatialTransformer
from keras.models import Sequential,Graph
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import f1_score
import sys
import pdb

size=int(sys.argv[1])
directory="train_photos/train"+sys.argv[1]+"/"
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))
def extract_images(x):
    x['random']=np.random.uniform(size=(x.shape[0],1))
    return(x.sort('random').head(10)['photo_id'].tolist())
def loadbusimage(x):
    bisimg=[]
    for f_ in x:
        img=imread(directory+str(f_)+".jpg").astype(np.float32)
        bisimg.append(np.array([img[:,:,0],img[:,:,1],img[:,:,2]]))
    for i in xrange(0,10-len(bisimg)):
        bisimg.append(np.array([img[:,:,0],img[:,:,1],img[:,:,2]]))
    try:
        return(np.array(bisimg))
    except:
        np.array(bisimg)
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
n_images=10
graph = Graph()
nfilters=32
for i in xrange(0,n_images):
    graph.add_input(name="input"+str(i),input_shape=(3,size,size))
graph.add_shared_node(Convolution2D(nfilters, 3, 3, border_mode='same',activation='relu'),name='conv1',inputs=["input"+str(i) for i in xrange(0,10)])
graph.add_shared_node(BatchNormalization(),name='batch1',inputs=['conv1'])
graph.add_shared_node(Convolution2D(nfilters,3,3,activation=LeakyReLU()), name='conv2', inputs=['batch1'])
graph.add_shared_node(BatchNormalization(),name='batch2',inputs=['conv2'])
graph.add_shared_node(Convolution2D(nfilters,3,3,activation=LeakyReLU()), name='conv3', inputs=['batch2'])
graph.add_shared_node(BatchNormalization(),name='batch3',inputs=['conv3'])
graph.add_shared_node(Convolution2D(nfilters,3,3,activation=LeakyReLU()), name='conv4', inputs=['batch3'])
graph.add_shared_node(BatchNormalization(),name='batch4',inputs=['conv4'])
graph.add_shared_node(Convolution2D(nfilters,3,3,activation=LeakyReLU()), name='conv5', inputs=['batch4'])
graph.add_shared_node(BatchNormalization(),name='batch5',inputs=['conv5'])
#graph.add_shared_node(MaxPooling2D(pool_size=(2,2)), name='maxpool', inputs=['conv5'])
graph.add_shared_node(Flatten(),name='flatten',inputs=['batch5'])
graph.add_shared_node(Dense(256,activation='relu'), name='dense1',inputs=['flatten'])
graph.add_shared_node(Dense(len(mlb.classes_),activation='sigmoid'),name='output1',inputs=['dense1'],merge_mode='ave',create_output=True)
#model.add(SpatialTransformer(localization_net=locnet,input_shape=(224, 224,30)))
sgd=SGD(lr=0.01,momentum=0.9,decay=1e-6,nesterov=True)
graph.compile(optimizer='adam',loss={'output1':'categorical_crossentropy'})
nfilters=40

#split into train/test sets, load in ten images per business
epochs=5000
#build average image
tphotos=train.groupby('business_id').apply(extract_images).reset_index()
tphotos.reindex(np.random.permutation(tphotos.index))
tphotos.columns=['business_id','photo_id']
tphotos=tphotos.merge(labels,on='business_id',how='left')
X_train=loadimages(tphotos['photo_id'])
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
#    X_train=np.random.uniform(size=(X_test.shape[0],3,224,224))
    pdb.set_trace()
    inputkeys={"input"+str(i):X_train[:,i,:,:,:] for i in xrange(0,n_images)}
    inputkeys['output1']=X_test
    graph.fit(inputkeys,nb_epoch=1,batch_size=16)
#    graph.fit({"input1":X_train,"input2":X_train,'output1':X_test},nb_epoch=2)
#    model.fit(X_train,X_test,batch_size=128,nb_epoch=1,verbose=0)
    inputkeys={"input"+str(i):Y_train[:,i,:,:,:] for i in xrange(0,n_images)}   
    prob=graph.predict(inputkeys)['output1']
    pred=np.round(prob)
#    probs=graph.predict_proba({"input1":Y_train[:,0,:,:,:],"input2":Y_train[:,1,:,:,:]})
#    print prob.mean(axis=0)
#    print prob.max(axis=0)
#    print prob.min(axis=0)
    print f1_score(Y_test,pred)
    print epoch
