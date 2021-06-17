import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import cv2
#import matplotlib.pyplot as plt
import math
from glob import glob
import os
import time
import progressbar

np.random.seed(6789)

start = time.time()


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


def load_path(path, cl, y, file_paths):
    
    max_len = len(os.listdir(path))
    bar = progressbar.ProgressBar(maxval=max_len, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    i = 0

    for p in os.listdir(path):
        #print (p)
        file_paths.append(path+p)
        y.append(cl)
        bar.update(i)

        i += 1

    bar.finish()    
    len(file_paths)
    return y, file_paths
    
def extract_features(X):
    i = 0
    features = []

    max_len = len(X)
    bar = progressbar.ProgressBar(maxval=max_len, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for x in X:
        #x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        #feature = model.predict(x)
        #flat = feature.flatten()
        features.append(x)#flat)
        #print (i)
        bar.update(i+1)
        i += 1

    bar.finish()
    #features = np.expand_dims(features, axis=0)
    
    return features
    

#load squeezenet model

#print (sys.argv[1])

#print ('Loading squeezenet model...')

img_rows, img_cols, img_channel = 50, 50, 3
#sq_layer = 'pool1'

#base_model = SqueezeNet(weights='imagenet', include_top=False,input_shape=(img_rows, img_cols, img_channel), pooling='avg')
#model = models.Model(inputs=base_model.input, outputs=base_model.get_layer(sq_layer).output)
start2 = time.time()
if True: #sys.argv[1] == 'ext_feat':

	print ('Extracting features from images..')

	#load file path names

	y = []
	file_paths = []

	num = 1300 #1400

	print ("Loading negative images")
	y, file_paths = load_path("./data/test_rotation5/", 0, y, file_paths)
	y = y[:num]
	file_paths = file_paths[:num]
	print ("Loading postive images")
	y, file_paths = load_path("./data/test_rotation11/", 1, y, file_paths)
	y = y[:num+num]
	file_paths = file_paths[:num+num]



	#shuffle data

	data_num = len(y)
	random_index = np.random.permutation(data_num)

	path_shuffle = []
	y_shuffle = []

	for i in range(data_num):
		path_shuffle.append(file_paths[random_index[i]])
		y_shuffle.append(y[random_index[i]])
		
	y = np.array(y_shuffle)



	#load images

	x = []

	max_len = len(path_shuffle)
	bar = progressbar.ProgressBar(maxval=max_len, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	i = 0

	for j, file_path in enumerate(path_shuffle):
		#read image
		##img = cv2.imread(file_path)
		#img = cv2.cvtColor(img)
		#print(img)
		
		img = image.load_img(file_path)
		x.append(image.img_to_array(img))
		bar.update(i)
		i+=1

	x = np.array(x)
	bar.finish()


	#label encoding

	y = to_categorical(y)


	#train-test split

	val_split_num = int(round(0.1*len(y)))
	x_train = x[val_split_num:]
	y_train = y[val_split_num:]
	x_test = x[:val_split_num] #x[:val_split_num]
	y_test = y[:val_split_num] #y[:val_split_num]

	print('x_train', x_train.shape)
	print('y_train', y_train.shape)
	print('x_test', x_test.shape)
	print('y_test', y_test.shape)



	#scale to range [0, 1]

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255



	#extract features

	train_features = extract_features(x_train)
	test_features = extract_features(x_test)

	timestamp = '{}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H %M %S'))

	train_features_path = "./data/features/train_features_{}.h5".format(timestamp)
	h5f_data = h5py.File(train_features_path, 'w')
	h5f_data.create_dataset('dataset_1', data=np.array(train_features))

	test_features_path = "./data/features/test_features_{}.h5".format(timestamp)
	h5f_data = h5py.File(test_features_path, 'w')
	h5f_data.create_dataset('dataset_1', data=np.array(test_features))

	train_labels = "./data/features/train_labels_{}.h5".format(timestamp)
	h5f_data = h5py.File(train_labels, 'w')
	h5f_data.create_dataset('dataset_1', data=np.array(y_train))

	test_labels = "./data/features/test_labels_{}.h5".format(timestamp)
	h5f_data = h5py.File(test_labels, 'w')
	h5f_data.create_dataset('dataset_1', data=np.array(y_test))


else:
	print ('Loading features from .h5 files..')	

	timestamp = 'pool1_2018-01-30 09 41 20.h5'

	train_features_path = './data/features/train_features_{}'.format(timestamp)
	test_features_path = './data/features/test_features_{}'.format(timestamp)
	train_labels = './data/features/train_labels_{}'.format(timestamp)
	test_labels = './data/features/test_labels_{}'.format(timestamp)



"""
print ('Loading data...')

data_path = './plots/plot_data_2018-02-05 05 01 52.h5'

hf = h5py.File(data_path, 'r')
train_features = np.array(hf.get('x_train'))

hf = h5py.File(data_path, 'r')
y_train = np.array(hf.get('y_train'))

hf = h5py.File(data_path, 'r')
test_features = np.array(hf.get('x_test'))

hf = h5py.File(data_path, 'r')
y_test = np.array(hf.get('x_test'))
"""

#training

batch_size = 128 #32, 64, 128, 256
learning_rate = 0.75e-5 #2e-4
epochs = 1000

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

pool1 = (2, 2)

model2 = models.Sequential()

model2.add(layers.Conv2D(32, kernel_size=[3, 3], strides=1, padding='same', input_shape = (img_rows, img_cols, img_channel)))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation('relu'))
model2.add(layers.MaxPooling2D(pool_size=pool1))
model2.add(layers.Dropout(0.25))

model2.add(layers.Conv2D(64, kernel_size=[3, 3], padding='same'))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation('relu'))
model2.add(layers.MaxPooling2D(pool_size=pool1))
model2.add(layers.Dropout(0.25))

model2.add(layers.GlobalAveragePooling2D())
model2.add(layers.Dense(10))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation('relu'))
model2.add(layers.Dense(2))
model2.add(layers.Activation('softmax'))

optimizer = 'rmsprop'

#model_pickle_path = './model_checkpoints/model_sn_2018-02-01 09 47 41.h5'#'./model_weights/model_sn_2018-01-3113 38 00.h5'

#model2 = load_model(model_pickle_path)

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

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./plots/accuracy_{}.png'.format(timestamp))
plt.show()


loss = history.history['loss']
val_loss = history.history['val_loss']

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
h5f_data.create_dataset('path_shuffle', data=np.array(path_shuffle))
h5f_data.create_dataset('y_shuffle', data=np.array(y_shuffle))
#cv2.waitkey(0)

end = time.time()

elapsed2 = end - start2
elapsed = end - start

print ('Total Elapsed Time: {} secs'.format(elapsed))
print ('Training Time: {} secs'.format(elapsed2))
