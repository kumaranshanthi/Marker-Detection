#sudo python TargetPS.py >> logfile.log 2>&1
#python /home/nvidia/Marker_test/Marker_Detection_latlong/Target_GPS.py >> logfile.log 2>&1
#sudo nano /home/nvidia/.bashrc
from time import sleep
from time import time
import datetime
import subprocess
import os
from datetime import datetime, timedelta
import pyexiv2
#import fractions
from fractions import Fraction
from PIL import Image
from PIL.ExifTags import TAGS
import sys
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil
#from ubidots import ApiClient
import sys
import time
import argparse 
import json
import random
import math
import serial
import socket, struct, time
#import pygame
#from modules.utils import *
import serial
import glob
import os



import cv2
import imutils
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


import math
import os
#.............
import glob
import os
from PIL import Image

import subprocess 

from time import sleep
from time import time
import datetime

from datetime import datetime, timedelta

import numpy, os, time, cv, sys, math, sys, glob, argparse
import multiprocessing


from keras.models import load_model
from keras.models import Sequential, Model, load_model
#from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image




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


#os.system('sudo ./sony.sh')
#subprocess.Popen("/home/nvidia/Marker_test/jeotag_code01/sony.sh", shell=True)
subprocess.Popen("/home/nvidia/Marker_test/Marker_Detection_latlong/sony.sh", shell=True)
# Connecting telemetry module
try:
	global vehicle

	#parser = argparse.ArgumentParser(description='Print out vehicle state information. Connects to SITL on local PC by default.')
	#parser.add_argument('--connect', 
		           #help="vehicle connection target string. If not specified, SITL automatically started and used.")
	#args = parser.parse_args()

	#connection_string = args.connect
	sitl = None


	#Start SITL if no connection string specified
	#if not connection_string:
	    #import dronekit_sitl
	    #sitl = dronekit_sitl.start_default()
	    #connection_string = sitl.connection_string()


	# Connect to the Vehicle. 
	#   Set `wait_ready=True` to ensure default attributes are populated before `connect()` returns.
	#print "\nConnecting to vehicle on: %s" % connection_string
	vehicle = connect("udp:192.168.1.102:14550", baud=57600, wait_ready=False)

	vehicle.wait_ready('autopilot_version')
except Exception, e:
   print "Error to Connect"

"""
UDP_IP = "127.0.0.1" # Localhost (for testing)
UDP_PORT = 51002 # This port match the ones using on other scripts
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#alt = 14
"""
#list_of_files = glob.glob('/home/pi/onboard-communication-pi/sony_image/*')



# Need to open the saved model object into read mode #2018-03-15 21 30 42.h5' #2018-03-08 09 20 42.h5'
model_pickle_path = './model_checkpoints/model_sn_2018-03-15 21 30 42.h5'  
#model_pickle_path = 'sudo /home/nvidia/Marker_test/Marker_Detection_latlong/model_checkpoints/model_sn_2018-03-15 21 30 42.h5' 
#model1 = models.load_model(model_pickle_path)
model1 = load_model(model_pickle_path)

print ('MODEL SUMMARY:')
model1.summary()

start = time.time()

path = './jeo_tag/'  
save_path = './data/bound_images/'

print ('\n\nDetecting markers...')

boundary = []
X = []


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


def take_picture():
    #os.system('sudo ./sony.sh')
    #subprocess.Popen("/home/nvidia/Marker_test/jeotag_code01/sony.sh", shell=True)
    subprocess.Popen("/home/nvidia/Marker_test/Marker_Detection_latlong/sony.sh", shell=True)
    sleep(2)
    #list_of_files = glob.glob('/home/pi/onboard-communication-pi/sony_image/*')
    list_of_files = glob.glob('/home/nvidia/Marker_test/Marker_Detection_latlong/jeo_tag/*')
    #GPIO.output(13,True)
    #os.system('sudo ./sony.sh')
    #print datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S') + " done"
    #GPIO.output(13,False)
    latest_file = max(list_of_files, key=os.path.getctime)
    print "latest_file", latest_file
    #sleep(2)
    return latest_file;

def set_arm(arm):
    if arm:
        print "Arming motors"
        vehicle.armed = True
    else:
        print "Disarming motors"
        vehicle.armed = False


def set_mode(mode):
    if mode == 1:
        vehicle.mode = VehicleMode("STABILIZE")

    if mode == 2:
        vehicle.mode = VehicleMode("ALT_HOLD")

    if mode == 3:
        vehicle.mode = VehicleMode("LOITER")

    if mode == 4:
        vehicle.mode = VehicleMode("RTL")

    if mode == 5:
        vehicle.mode = VehicleMode("AUTO")

    print "Change mode: " + str(mode)

def takeoff(t):
    vehicle.commands.takeoff(t)
    print "takeoff at " + str(t)
 
# Function to convert data to voltage level,
# rounded to specified number of decimal places.
 
# END: GENERAL FUNCTIONS

def to_deg(value, loc):
  if value < 0:
    loc_value = loc[0]
  elif value > 0:
    loc_value = loc[1]
  else:
    loc_value = ""
  abs_value = abs(value)
  #abs_value = (value)
  deg =  int(abs_value)
  #deg =  (abs_value)
  t1 = (abs_value-deg)*60
  min = int(t1)
  sec = round((t1 - min)* 60, 5)
  return (deg, min, sec, loc_value)



def set_gps_location(latest_file, lat, lng, alt):
    """
    
    
    Adds GPS position as EXIF metadata

    Keyword arguments:
    file_name -- image file 
    lat -- latitude (as float)
    lng -- longitude (as float)

    """

    lat_deg = to_deg(lat, ["S", "N"])
    lng_deg = to_deg(lng, ["W", "E"])

    # convert decimal coordinates into degrees, munutes and seconds
    exiv_lat = (pyexiv2.Rational(lat_deg[0]*60+lat_deg[1],60),
                pyexiv2.Rational(lat_deg[2]*100,6000),
                pyexiv2.Rational(0, 1))
    exiv_lng = (pyexiv2.Rational(lng_deg[0]*60+lng_deg[1],60),
                pyexiv2.Rational(lng_deg[2]*100,6000),
                pyexiv2.Rational(0, 1))

    m = pyexiv2.ImageMetadata(latest_file)
    m.read()

    m["Exif.GPSInfo.GPSLatitude"] = exiv_lat
    m["Exif.GPSInfo.GPSLatitudeRef"] = lat_deg[3]
    m["Exif.GPSInfo.GPSLongitude"] = exiv_lng
    m["Exif.GPSInfo.GPSLongitudeRef"] = lng_deg[3]
    m["Exif.GPSInfo.GPSAltitude"] = Fraction(int(alt))
    m["Exif.Image.GPSTag"] = 654
    m["Exif.GPSInfo.GPSMapDatum"] = "WGS-84"
    m["Exif.GPSInfo.GPSVersionID"] = '2 0 0 0'
    #m["Exif.Image.DateTime"] = datetime.datetime.fromtimestamp(t)
    m.write()


#path = './jeo_tag/'  
#save_path = './data/bound_images/'
#time.sleep(100)
#print "subprocess program start"
#subprocess.Popen("python /home/nvidia/Marker_test/jeotag_code01/app_basics5.py", shell=True) 
#os.system('python /home/nvidia/Marker_test/jeotag_code01/app_basics5.py')
time.sleep(1)
#while True:
for j, p in enumerate(sorted(os.listdir(path))[:]): #[100:700]:
                ############################
                lat = vehicle.location.global_relative_frame.lat
		lon = vehicle.location.global_relative_frame.lon
		alt = vehicle.location.global_relative_frame.alt
		alt = abs(alt)
		arm = str(vehicle.armed)
		mode = str(vehicle.mode.name)
		#batt = str(vehicle.battery.level
		print "muthu"
		time.sleep(2)
		    
		print "Mode: %s" % mode
		print "Latitud: ", (lat)
		print "Longitud: ", (lon)
		print "Attitude: ", (alt)
		#latest_file = take_picture()
		#set_gps_location(latest_file, lat, lon, alt)
                ################################
                print "muthuselvam"
      		#list_of_files = glob.glob('/home/nvidia/Marker_test/Marker_Detection_latlong/jeo_tag/*')
                list_of_files = glob.glob('/home/nvidia/Marker_test/Marker_Detection_latlong/jeo_tag02/*')
	        f = max(list_of_files, key=os.path.getctime) #'DSC01251.JPG'
                print "sel"
	        time.sleep(1)

	        #pos = mav_position.exif_position(f)
	        #pos = exif_position(f)
		#h, w, ch = img.shape
		img_shape = (3376, 6000)


		pos = mav_position.exif_position(f)
                print "sel1"
		print ("alt", pos.altitude)
		C_params = cam_params.CameraParams(lens=15.6, sensorwidth=23.5)



                ####################################
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
                        ####################################
			#x = 2000
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
                                """
		                message01 = str(lat) + " " + str(lon)
		    		print (type(message01)) 
		    		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	   	    		server_address = ('192.168.1.3', 10005)
		    		#print >>sys.stderr, 'sending "%s"' % message01
		    		sent = sock.sendto(message01, server_address)
		    		print (message01)
				"""

