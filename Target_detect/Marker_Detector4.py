import cv2
import numpy as np 
import os
import time
import pandas as pd
import datetime
import math
import pickle
import progressbar

import matplotlib.pyplot as plt
from PIL import Image

import subprocess 

from keras.models import load_model
from keras.models import Sequential, Model, load_model
#from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

#import cam_params
#from gps_coordinate import exif_position, gps_position_from_xy
#...... ......................
#import cam_params
###from gps_coordinate import exif_position

import math
import os
#.............
import glob
import os
from PIL import Image

from time import sleep
from time import time
import datetime
import os
from datetime import datetime, timedelta

import numpy, os, time, cv, sys, math, sys, glob, argparse
import multiprocessing

from cuav.lib import cuav_util
#from cuav.image import scanner
from cuav.lib import mav_position, cuav_joe, cuav_region
from cuav.camera import cam_params
from MAVProxy.modules.mavproxy_map import mp_slipmap
from MAVProxy.modules.lib import mp_image
from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
from gooey import Gooey, GooeyParser

slipmap = None
mosaic = None
#................................

import socket
import sys

"""

while True:
	#Create a TCP/IP socket
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

	# Bind the socket to the port
	server_address = ('', 10005)
	print 'starting up on %s port %s' % server_address
	sock.bind(server_address)

	print 'waiting to receive message'
	#data, address = sock.recvfrom(4096)
	data, address = sock.recvfrom(1024)

	print 'received %s bytes from %s' % (len(data), address)
	print data

	f = open('camera.txt','w')
	f.write(data)
	f.close()
	f=open('/home/muthu/7.4.18_collision_03/Marker_Detection_latlong/camera.txt')
	lines=f.readlines()
	#print lines
	result=[]
	for x in lines:
		result.append(x.split(' ')[0])
		result.append(x.split(' ')[1])
	joe_lat_search = float(result[0])
	joe_lon_search= float(result[1])
	#print "goe_lat", joe_lat
	#print "goe_lon", joe_lon
	f.close() 
	dat = len(str(joe_lat_search))
	if dat >= 6:
		print "camera Trigger started" 
                f = open('camera.txt','w')
	        sel = ("0 0")
	        f.write(sel)
	        f.close
		break

"""
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
	
	
def extract_marker(path, save_path):
	boundary = []
	X = []
        count = 0
        """
        while True:
                count = count+1
                print "count", count
                if count == 3:
                        print "subprocess program start"
                	subprocess.Popen("python /home/nvidia/Marker_test/jeotag_code01/app_basics5.py", shell=True)     
        """  
	for j, p in enumerate(sorted(os.listdir(path))[:]): #[100:700]: while True: #
		p = sorted(os.listdir(path))[-1]
		boundary = []
		X = []
		
		print (path+p)

		img = cv2.imread(path+p)
		
		if img is None:
                        print "muthu"
			continue
	
		h, w, ch = img.shape
		img_shape = (h, w)

		lower_white = np.array([160,160,160])
		upper_white = np.array([255,255,255])
	
		mask = cv2.inRange(img, lower_white, upper_white)
	
		res = cv2.bitwise_and(img,img, mask = mask)
	
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #_, 

		contour_list = []
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > 100:
				contour_list.append(contour)


		vertices = []
		for c in contour_list:
			vertices.append(find_vertices(c, h, w))
	
		#img2 = cv2.imread(path+p)
                print "sel"
	
		for i, v in enumerate(vertices):
                        #print "messi"
			crop_img = img[v[0][1]:v[1][1], v[0][0]:v[1][0]] # img[y:y+h, x:x+w]
			shape = crop_img.shape
			#cv2.rectangle(img2, v[0], v[1], (255,0,0), 2)
			#print (v)
			
			crop_file = save_path + p[:-4] + "_" + str(i).zfill(4) + ".png"
			
			#resized = imutils.resize(crop_img, width=50, height=50)
			resized = cv2.resize(crop_img, (50, 50), interpolation = cv2.INTER_AREA)
			#print (resized.shape)
			x = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
			x = x.astype('float32')
			#print (x.shape)
			X.append(x)
			#cv2.imwrite(crop_file, resized)
		
			boundary.append([p, v])

		#print ("Predicting...")
		X = np.array(X)
		
		if 0 in X.shape:
			continue
		
		X = X.astype('float32')
		X /= 255
		preds = model1.predict(X)
	
		print (preds.shape)
	
		preds = np.argmax(preds, axis=1)

		df = pd.DataFrame(boundary, columns=['file_name','vertices'])
		df.loc[:,'pred'] = pd.Series(preds, index=df.index)
		df2 = df[df.pred==1]
				#print (df)
		print ("Saving images..")	
		for index, row in df2.iterrows():
		   filepath2 = row['file_name']
		   #print (filepath2)
		   pred = row['pred']
		   v = row['vertices']
		   if pred == 1:
			    print (path+filepath2)
			    #pos = exif_position(path+filepath2)
                            pos = mav_position.exif_position(path+filepath2)
                            print ("alt", pos.altitude)
			    C_params = cam_params.CameraParams(lens=15.6, sensorwidth=23.5)
			    x = (v[0][0]+v[1][0])/2.0
			    y = (v[0][1]+v[1][1])/2.0
                            print "x,y", (x,y)
                            #x = 3000
                            #y = 1688
			   #latlon = gps_position_from_xy(x, y, pos, C=C_params, shape=img_shape)
                            latlon = cuav_util.gps_position_from_xy(x, y, pos, C=C_params, shape=img_shape)
			   #center = gps_position_from_xy(img_shape[0]/2, img_shape[1]/2, pos, C=C_params, shape=img_shape)
			    message = str(latlon[0]) + " " + str(latlon[1]) #str(latlon.lat) + " " + str(latlon.lon)
	                    lat=latlon[0]
			    lon=latlon[1]
			    xarray = numpy.array([lat])
			    yarray = numpy.array([lon])
			    #here is your data, in two numpy arrays
			    data = numpy.array([xarray, yarray])
			    data = data.T
			    #here you transpose your data, so to have it in two columns
			    print "sssssssssssssssssssssssssssssssssssss"
			    datafile_path = "/home/nvidia/Marker_test/Marker_Detection_latlong/geo01.txt"
			    with open(datafile_path, 'w+') as datafile_id:
			       #here you open the ascii file

		    	       numpy.savetxt(datafile_id, data, fmt=['%f','%f'])
			    print ('Marker at [{}]'.format(message))
			    #print ('UAV location: {}'.format(center))
                            print "uav loc", (pos.lat, pos.lon)
			   
			    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			    #server_address = ('localhost', 10000)
                            server_address = ('192.168.98.87', 10005)
    		   print >>sys.stderr, 'sending "%s"' % message
    		   sent = sock.sendto(message, server_address)
    		   sock.close()
    		   return
			   

#subprocess.Popen("sudo python /home/nvidia/Marker_test/jeotag_code/app_basics5.py", shell=True)

# Need to open the saved model object into read mode #2018-03-15 21 30 42.h5' #2018-03-08 09 20 42.h5'
model_pickle_path = './model_checkpoints/model_sn_2018-03-15 21 30 42.h5'  
#model_pickle_path = 'sudo /home/nvidia/Marker_test/Marker_Detection_latlong/model_checkpoints/model_sn_2018-03-15 21 30 42.h5' 
#model1 = models.load_model(model_pickle_path)
model1 = load_model(model_pickle_path)

print ('MODEL SUMMARY:')
model1.summary()

start = time.time()

path = './jeo_tag02/'  
save_path = './data/bound_images/'

print ('\n\nDetecting markers...')

extract_marker(path, save_path)

end = time.time()
elapsed = end - start

print ('Total Elapsed Time: {} secs'.format(elapsed))

