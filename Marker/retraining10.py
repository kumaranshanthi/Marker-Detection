import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import cv2
import matplotlib.pyplot as plt
import math
from glob import glob
import os
import progressbar

np.random.seed(6789)


from keras.models import Sequential, Model, load_model, model_from_json
#from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
#from keras.layers import Input, Dense
#from keras import optimizers
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers


#import pickle #as cPickle

#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


import h5py
import time
import datetime
import sys



#print (sys.argv[1])

#print ('Loading squeezenet model...')

img_rows, img_cols, img_channel = 50, 50, 3

timestamp = '{}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H %M %S'))
prev_timestamp = '2018-04-02 13 32 28' #'2018-02-28 09 17 18' #'2018-02-05 12 30 25'

print ('Loading data...')

data_path = './plots/plot_data_{}.h5'.format(prev_timestamp) #plot_data_2018-02-28 09 17 18

print (data_path)

hf = h5py.File(data_path, 'r')
train_features = np.array(hf.get('train_features'))

hf = h5py.File(data_path, 'r')
y_train = np.array(hf.get('y_train'))

hf = h5py.File(data_path, 'r')
test_features = np.array(hf.get('test_features'))

hf = h5py.File(data_path, 'r')
y_test = np.array(hf.get('y_test'))


#training

batch_size = 128
learning_rate = 0.75e-5 #2e-4
epochs = 10

print ('batch_size: ' + str(batch_size))
print ('learning_rate: ' + str(learning_rate))
print ('epochs: ' + str(epochs))

train_features = np.array(train_features)
#train_features = np.expand_dims(train_features, axis=0)
input_shape = (None, img_rows, img_cols, img_channel)
print (input_shape)

test_features = np.array(test_features)
#test_features = np.expand_dims(test_features, axis=0)
print (test_features.shape)


optimizer = 'rmsprop'

model_pickle_path = './model_checkpoints/model_sn_{}.h5'.format(prev_timestamp) #'./model_weights/model_sn_2018-01-3113 38 00.h5'

model2 = load_model(model_pickle_path)

model2.compile(optimizer=optimizers.rmsprop(lr=learning_rate), #adam, rmsprop, sgd
              loss='categorical_crossentropy',
              metrics=['acc'])

model2.summary()

history = model2.fit(train_features,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(test_features,y_test))

def compute_accuracy(y_test, preds):
    l = len(y_test)
    count = 0

    for i in range(l):
        if(y_test[i] == preds[i]):
            count += 1

    accuracy = count*100//l
    
    return accuracy
    
#preds = model.predict(test_features)
#accuracy1 = compute_accuracy(y_test, preds)
#print('Accuracy: ' + str(accuracy1))

#print("{}\n".format(classification_report(y_test, preds)))

model_pickle_path = './model_checkpoints/model_sn_{}.h5'.format(timestamp)
#model_pickle = open(model_pickle_path, 'wb')
#pickle.dump(model, model_pickle)
#model_pickle.close()

model2.save(model_pickle_path)

#plot accuracy

acc = np.array(hf.get('acc'))
acc = np.append(acc, history.history['acc'])
val_acc = np.array(hf.get('val_acc'))
val_acc = np.append(val_acc, history.history['val_acc'])

plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./plots/accuracy_{}.png'.format(timestamp))
plt.show()


#plot loss

loss = np.array(hf.get('loss'))
loss = np.append(loss, history.history['loss'])
val_loss = np.array(hf.get('val_loss'))
val_loss = np.append(val_loss, history.history['val_loss'])

plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./plots/loss_{}.png'.format(timestamp))
plt.show()


tag = 'lr{}_{}_ep{}_bat{}_{}'.format(learning_rate, optimizer, epochs, batch_size, timestamp)

plot_data_path = './plots/plot_data_{}.h5'.format(timestamp)
h5f_data = h5py.File(plot_data_path, 'w')
h5f_data.create_dataset('acc', data=np.array(acc))
h5f_data.create_dataset('val_acc', data=np.array(val_acc))
h5f_data.create_dataset('loss', data=np.array(loss))
h5f_data.create_dataset('val_loss', data=np.array(val_loss))

h5f_data.create_dataset('train_features', data=np.array(train_features))
h5f_data.create_dataset('y_train', data=np.array(y_train))
h5f_data.create_dataset('test_features', data=np.array(test_features))
h5f_data.create_dataset('y_test', data=np.array(y_test))
#cv2.waitkey(0)
