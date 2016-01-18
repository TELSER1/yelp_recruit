import pandas as pd
import numpy as np
import os
import pdb
from scipy.misc import imread
from sklearn.metrics import log_loss
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

batch_size=64
def isclassthere(x,y):
    if y in x:
        return 1
    else:
        return 0
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

def load_data():
    labels=pd.read_csv("train.csv")
    bismatch=pd.read_csv("train_photo_to_biz_ids.csv")
    labels=bismatch.merge(labels,how='left',on='business_id')
    labels=labels[pd.isnull(labels['labels'])==False]
    labels['labels']=labels['labels'].map(lambda x:[int(i) for i in x.split(" ")])
    training_=os.listdir("train_photos/train244")
    train_ids=pd.DataFrame({"photo_id":[int(i.split(".")[0]) for i in training_]})
    train_ids=train_ids.merge(labels,on='photo_id',how='inner')
#    val_ids=val_ids.merge(labels,on='photo_id',how='inner')
    mlb=MultiLabelBinarizer()
    mlb.fit(train_ids['labels'].tolist())
#    X_train=np.array([imread('train_photos/train244/'+str(f_)+".jpg") for f_ in train_ids['photo_id'].tolist()]).astype(np.float32)
#    X_test=np.array([imread('train_photos/val244/'+str(f_)+".jpg") for f_ in val_ids['photo_id'].tolist()]).astype(np.float32)
    return train_ids,mlb
def load_train(train_list):
    return(np.array([imread('train_photos/train244/'+str(f_)+".jpg") for f_ in train_list]).astype(np.float32))
train_ids,mlb=load_data()
train_ids['assignment']=np.random.uniform(size=(train_ids.shape[0],1))
val_ids=train_ids[train_ids['assignment']>=.9].reset_index(drop=True)
Y_test=mlb.transform(val_ids['labels'].tolist())
print Y_test.shape
np.random.seed(42)
#train_ids=train_ids.sort('business_id').reset_index(drop=True)
train_ids.reindex(np.random.permutation(train_ids.index))
val_ids.reindex(np.random.permutation(val_ids.index))
validate=np.array([imread('train_photos/train244/'+str(f_)+".jpg") for f_ in val_ids['photo_id'].tolist()[0:10000]]).astype(np.float32)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    horizontal_flip=True)
X_train=load_train(train_ids['photo_id'][0:10000].tolist())
datagen.fit(X_train)
del X_train
train_ids=train_ids.sort('business_id').reset_index(drop=True)
validate=np.array([imread('train_photos/train244/'+str(f_)+".jpg") for f_ in val_ids['photo_id'].tolist()]).astype(np.float32)
model=Sequential()
model.add(Convolution2D(32, 3, 3,border_mode='valid',dim_ordering='tf',input_shape=(224, 224,3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(9))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
n_epoch=40
increment=8000
for epoch in xrange(n_epoch):
    print epoch
    bindex=0
    iter_=True
    while iter_==True:
        Y_train=mlb.transform(train_ids['labels'][bindex:bindex+increment].tolist())
        X_train=load_train(train_ids['photo_id'][bindex:bindex+increment].tolist())
        print X_train.shape
        for X_batch,_batch in datagen.flow(X_train,Y_train):
            model.fit(X_train,Y_train,batch_size=128,nb_epoch=1)
              
        if bindex>train_ids.shape[0]:
            print bindex
            iter_=False
        bindex+=increment
        print log_loss(Y_test,model.predict_proba(validate))
    #train_ids.reindex(np.random.permutation(train_ids.index))
