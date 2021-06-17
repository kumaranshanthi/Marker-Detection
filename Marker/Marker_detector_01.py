#!/usr/bin/python
import glob
import cv2
import numpy as np 
import os
import time
import pandas as pd
import datetime

import numpy, os, time, sys, math, sys, glob, argparse

import matplotlib.pyplot as plt
from PIL import Image

from keras.models import Sequential, Model, load_model
#from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

import math
import pickle
import os
import time
import progressbar

def find_vertices(contour, h, w):
	
	min_x = 10000
	min_y = 10000
	max_x = 0
	max_y = 0
	
	gap = 1
	
	for c in contour:

		x = c[0][0]
		y = c[0][1]
		
		if x < 0:
			x += h
			
		if y < 0:
			y += w		
		
		if x < min_x:
			min_x = x
			
		if y < min_y:
			min_y = y
			
		if x > max_x:
			max_x = x
			
		if y > max_y:
			max_y = y
	
	min_x -= gap
	min_y -= gap
	max_x += gap
	max_y += gap
	
	min_x = max(min_x, 0)
	min_y = max(min_y, 0)
	max_x = min(max_x, w)
	max_y = min(max_y, h)

	return [(min_x, min_y), (max_x, max_y)]
	
path = './data/landscape_set5/*' #/DownSampledImages/' #'./data/landscape_set5/' 
save_path = './data/cropped_images/'

print "while_loop"	
boundary = []
print "selllll01"
while True:
                list_of_files = glob.glob(path)
      		p = max(list_of_files, key=os.path.getctime)
      		time.sleep(1)
		print "sellll02"
		print (path+p)

		img = cv2.imread(path+p)
		
		if img is None:
			continue
	
		h, w, ch = img.shape
		#print (h, w)
		print (p)	

		lower_white = np.array([160,160,160])
		upper_white = np.array([255,255,255])
	
		mask = cv2.inRange(img, lower_white, upper_white)
	
		res = cv2.bitwise_and(img,img, mask = mask)
		#cv2.imwrite(save_path + 'res_' + p, res)	
	
		_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #_, 

		contour_list = []
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > 100:
				contour_list.append(contour)


		vertices = []
		for c in contour_list:
			vertices.append(find_vertices(c, h, w))
	        print "vertices", vertices
		#img2 = cv2.imread(path+p)
	        print "sellll03"
		for i, v in enumerate(vertices):
                        print "sellllll04"
			crop_img = img[v[0][1]:v[1][1], v[0][0]:v[1][0]] # img[y:y+h, x:x+w]
			shape = crop_img.shape
			#cv2.rectangle(img2, v[0], v[1], (255,0,0), 2)
			#print (v)
			
			crop_file = save_path + p[:-4] + "_" + str(i).zfill(4) + ".png"
			
			cv2.imwrite(crop_file, crop_img)
			"""
			img3 = Image.open(path + p)
			img3 = img3.resize((50, 50), Image.ANTIALIAS) #resize to 50x50
			img3.save(crop_file)
			"""
			#print (p, crop_file)
			#cv2.imwrite(crop_file, crop_img)
		
			boundary.append([crop_file, v])
		

		#cv2.imwrite(save_path + 'bounding_box_' + p, img2)
	
                print "selllll05"
	        df = pd.DataFrame(boundary, columns=['file_name','vertices'])
	        print (df.shape)




path = './data/landscape_set5/' #/DownSampledImages/' #'./data/landscape_set5/' 
save_path = './data/cropped_images/'

print ('\n\nDetecting objects...')

extract_marker(path, save_path)

#print (preds)

#print ('\n\nNegative:')
#dir_path = './data/test_rotation5/' #original_non_marker_images/' # neg_obj/
#preds = predict_dir(dir_path)
#print (preds)

