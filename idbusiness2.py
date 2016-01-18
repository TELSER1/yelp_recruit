import pandas as pd
import numpy as np
import os
import pdb
from scipy.misc import imread
from sklearn.metrics import log_loss
from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer,LabelEncoder
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
batch_size=64
def isclassthere(x,y):
    if y in x:
        return 1
    else:
        return 0
def gen_pred(predictions,BETA,nneighbors):
    pred=[]
    for i in xrange(0,predictions.shape[0]):
        ind=np.argpartition(predictions[i,:],-nneighbors)[-nneighbors:]
        cat=[]
        for category in xrange(0,9):
            cat.append(np.round(BETA[ind,category].sum()/(1.0*nneighbors)))
        pred.append(cat)
    return(np.array(pred))

    return
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
    mlb=LabelEncoder()
    mlb.fit(train_ids['business_id'].tolist())
#    X_train=np.array([imread('train_photos/train244/'+str(f_)+".jpg") for f_ in train_ids['photo_id'].tolist()]).astype(np.float32)
#    X_test=np.array([imread('train_photos/val244/'+str(f_)+".jpg") for f_ in val_ids['photo_id'].tolist()]).astype(np.float32)
    return train_ids,mlb
def load_train(train_list):
    return(np.array([imread('train_photos/train244/'+str(f_)+".jpg") for f_ in train_list]).astype(np.float32)/255.0)
train_ids,mlb=load_data()
labels=pd.read_csv("train.csv")
labels=labels[pd.isnull(labels['labels'])==False].reset_index(drop=True)
labels['assignment']=np.random.uniform(size=(labels.shape[0],1))

MLB=MultiLabelBinarizer()
train_ids=train_ids.merge(labels[['business_id','assignment']],on='business_id',how='left')
MLB.fit(train_ids['labels'].tolist()) 
labels['labels']=labels['labels'].map(lambda x:[int(i) for i in x.split(" ")])
BETA=MLB.transform(labels.sort('business_id')['labels'])
val_ids=train_ids[train_ids['assignment']>=.9].reset_index(drop=True)
val_Y=MLB.transform(val_ids['labels'])
train_ids=train_ids[train_ids['assignment']<.9].reset_index(drop=True)
Y_test=mlb.transform(val_ids['business_id'].tolist())
print Y_test.shape
np.random.seed(42)
#train_ids=train_ids.sort('business_id').reset_index(drop=True)
train_ids.reindex(np.random.permutation(train_ids.index))
val_ids.reindex(np.random.permutation(val_ids.index))
validate=np.array([imread('train_photos/train244/'+str(f_)+".jpg") for f_ in val_ids['photo_id'].tolist()[0:10000]]).astype(np.float32)/255.0

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    horizontal_flip=True)
X_train=load_train(train_ids['photo_id'][0:10000].tolist())
datagen.fit(X_train)
del X_train
#train_ids=train_ids.sort('business_id').reset_index(drop=True)
train_ids.reindex(np.random.permutation(train_ids.index))
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
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(len(mlb.classes_)))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
n_epoch=40
increment=1000

for epoch in xrange(n_epoch):
    print epoch
    bindex=0
    iter_=True
    while iter_==True:
        try:
            Y_train=mlb.transform(train_ids['business_id'][bindex:bindex+increment].tolist())
            X_train=load_train(train_ids['photo_id'][bindex:bindex+increment].tolist())
            print X_train.shape
            for X_batch,Y_batch in datagen.flow(X_train,Y_train,batch_size=increment):
                model.fit(X_batch,Y_batch,batch_size=128,nb_epoch=1)
            if bindex>train_ids.shape[0]:
                iter_=False
#            predictions=pd.DataFrame(model.predict_proba(validate))
#            predictions['business_id']=validate['business_id']
#            pgroup=predictions.groupby('business_id').reset_index()
            print bindex
            bindex+=increment
            iter_=False
        except:
            iter_=False
    proba=model.predict_proba(validate)
    #predictions=np.vstack([np.transpose((proba[i,:]*np.transpose(BETA)).sum(axis=1)) for i in xrange(0,proba.shape[0])])/1996.0

    pred2=gen_pred(proba,BETA,100)
    #predictions=np.round(predictions)
    #print predictions.sum()
    print f1_score(val_Y,pred2)
    train_ids.reindex(np.random.permutation(train_ids.index))
