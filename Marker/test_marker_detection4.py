import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from keras.models import Sequential, Model, load_model
#from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

import cv2

import math
#import pickle
import os
import time
import progressbar


def predict_dir(path):

	start = time.time()
	
	max_len = len(os.listdir(path))
	bar = progressbar.ProgressBar(maxval=max_len, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	i = 0
	X = []
	
	path_list = sorted(os.listdir(path))
	
	for p in path_list:
		img_path = path + p

		#img = Image.open(img_path)
		#img = img.resize((50, 50), Image.ANTIALIAS) #resize to 50x50
		#img.save('temp.jpg')

		#pre processing
		img = image.load_img(img_path) #'temp.jpg')
		x = image.img_to_array(img)
		
		X.append(x)
		bar.update(i)
		i += 1
		
	X = np.array(X)
	X = X.astype('float32')
	X /= 255
	preds = model1.predict(X)
	preds = np.round(preds)
	preds = preds.astype('int')
	pred_sum = preds.sum(axis=0)
	
	for i, p in enumerate(path_list):
		#print (p)
		img = cv2.imread(path + p)
		
		if preds[i][0] == 1:
			save_path = './data/class_0/' + p
		else:
			save_path = './data/class_1/' + p
			
		cv2.imwrite( save_path, img)
		#print (save_path)
	
	neg_acc = float(pred_sum[0])/(pred_sum[0] + pred_sum[1])
	pos_acc = float(pred_sum[1])/(pred_sum[0] + pred_sum[1])
	bar.finish()
	end = time.time()
	elapsed = end - start
	
	print (pred_sum)
	print ('Negative: {}%'.format(neg_acc*100))
	print ('Positive: {}%'.format(pos_acc*100))
	print ('Elapsed Time: {} secs'.format(elapsed))
	
	return preds


# Need to open the pickled model object into read mode
model_pickle_path = './model_checkpoints/model_sn_2018-03-28 18 49 30.h5' #2018-03-15 21 30 42.h5' #model_sn_2018-03-08 09 20 42.h5' #2018-02-28 06 33 02.h5'#model_sn_2018-02-05 10 22 55.h5'# model_sn_2018-02-05 10 22 55.h5 # model_sn_2018-02-05 06 47 26.h5

model1 = load_model(model_pickle_path)

print ('MODEL1:')
model1.summary()

img_rows, img_cols, img_channel = 50, 50, 3

print ('\n\nPositive:')
dir_path = './data/test_rotation4/' #original_non_marker_images/' # neg_obj/
preds = predict_dir(dir_path)
#print (preds)

print ('\n\nNegative:')
dir_path = './data/test_rotation5/' #original_non_marker_images/' # neg_obj/
preds = predict_dir(dir_path)
#print (preds)

