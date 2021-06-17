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

import argparse
import cv2
import serial
import os
import math
import socket, struct, time
import pygame
from modules.utils import *
import time


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




def take_picture():
    os.system('sudo ./sony.sh')
    #subprocess.Popen("/home/nvidia/Marker_test/jeotag_code01/sony.sh", shell=True)
    #subprocess.Popen("/home/nvidia/Marker_test/Marker_Detection_latlong/sony.sh", shell=True)
    #sleep(2)
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
while True:
                ############################
        lat = vehicle.location.global_relative_frame.lat
	lon = vehicle.location.global_relative_frame.lon
	alt = vehicle.location.global_relative_frame.alt
	alt = abs(alt)
	arm = str(vehicle.armed)
	mode = str(vehicle.mode.name)
	#batt = str(vehicle.battery.level
	print "muthu"
	#time.sleep(2)
	    
	print "Mode: %s" % mode
	print "Latitud: ", (lat)
	print "Longitud: ", (lon)
	print "Attitude: ", (alt)
	latest_file = take_picture()
	set_gps_location(latest_file, lat, lon, alt)
        ################################
        print "muthuselvam"
	list_of_files = glob.glob('/home/nvidia/Marker_test/Marker_Detection_latlong/jeo_tag/*')
        f = max(list_of_files, key=os.path.getctime) #'DSC01251.JPG'
        print "sel"
        #time.sleep(1)

        #pos = mav_position.exif_position(f)
        #pos = exif_position(f)
	#h, w, ch = img.shape
	img_shape = (3376, 6000)


	pos = mav_position.exif_position(f)
        print "sel1"
	print ("alt", pos.altitude)
	C_params = cam_params.CameraParams(lens=15.6, sensorwidth=23.5)



                ####################################
	status = "No Targets"

	# check to see if we have reached the end of the
	# video
	"""
	if not grabbed:
		break
	"""
	# convert the frame to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	edged = cv2.Canny(blurred, 50, 150)

	# find contours in the edge map
	(_, cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.imshow("contour", edged)
	# loop over the contours
	total = 0
	
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)

		# ensure that the approximated contour is "roughly" rectangular
		if len(approx) >= 4 and len(approx) <= 6:
			# compute the bounding box of the approximated contour and
			# use the bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			aspectRatio = w / float(h)
                        #print "muthu"
			# compute the solidity of the original contour
			area = cv2.contourArea(c)
			hullArea = cv2.contourArea(cv2.convexHull(c))
			solidity = area / float(hullArea)

			# compute whether or not the width and height, solidity, and
			# aspect ratio of the contour falls within appropriate bounds
			keepDims = w > 15 and h > 15
			keepSolidity = solidity > 0.9
                        #keepSolidity = solidity > 0.003
			keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2
                        
                        #print "width", w
                        #print "height", h
                        #print "solidity", solidity
                        #print "aspectRatio", aspectRatio
			# ensure that the contour passes all our tests
			if keepDims and keepSolidity and keepAspectRatio:
				# draw an outline around the target and update the status
				# text
				cv2.drawContours(frame, [approx], -1, (0, 255, 0), 4)
                                total += 1
				status = "Target(s) Acquired"
                                #print "I found {0} books in that image".format(total)
                                print "Number of Square", total
				# compute the center of the contour region and draw the
				# crosshairs
				M = cv2.moments(approx)
				(x, y) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				(startX, endX) = (int(x - (w * 0.15)), int(x + (w * 0.15)))
				(startY, endY) = (int(y - (h * 0.15)), int(y + (h * 0.15)))
				cv2.line(frame, (startX, y), (endX, y), (0, 0, 255), 3)
				cv2.line(frame, (x, startY), (y, endY), (0, 0, 255), 3)
                                #center = (cX, cY)
        			message = [x, y]
                                print message
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
		                        
				        message01 = str(lat) + " " + str(lon)
			    		print (type(message01)) 
			    		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		   	    		server_address = ('192.168.1.3', 10005)
			    		#print >>sys.stderr, 'sending "%s"' % message01
			    		sent = sock.sendto(message01, server_address)
			    		print (message01)
				

