# Python program to detect white color and save the top left and 
# bottom right corners of ROI

import cv2
import numpy as np 
import os
import time
import pandas as pd
import datetime


def find_vertices(contour, h, w):
	
	min_x = 10000
	min_y = 10000
	max_x = 0
	max_y = 0
	
	gap = 1 #15 in case rotation has to be done (after rotation, run this code again with gap = 1), else 1
	
	for c in contour:
		
		#print (c[0])
		x = c[0][0]
		y = c[0][1]
		
		#print (x)
		#print (y)
		
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
		
	#print ("min_x: " + str(min_x))
	#print ("max_x: " + str(max_x))
	
	min_x -= gap
	min_y -= gap
	max_x += gap
	max_y += gap
	
	#print ("min_x: " + str(min_x))
	#print ("max_x: " + str(max_x))
	
	#print ("\n")
	
	
	min_x = max(min_x, 0)
	min_y = max(min_y, 0)
	max_x = min(max_x, w)
	max_y = min(max_y, h)
	

	return [(min_x, min_y), (max_x, max_y)]
	
def extract_vertices(contours):
	
	vertices = []
	
	for c in contours:
		print (c[0][0])
		vertices.append(c[0][0])

	return vertices	

def extract_marker(path, save_path):
	boundary = []

	for p in sorted(os.listdir(path)): #[100:700]:

		img = cv2.imread(path+p)
	
		h, w, ch = img.shape
		#print (h, w)
		print (p)	

		lower_white = np.array([170,170,170])
		upper_white = np.array([255,255,255])
	
		mask = cv2.inRange(img, lower_white, upper_white)
	
		res = cv2.bitwise_and(img,img, mask= mask)
		#cv2.imwrite(save_path + 'res_' + p, res)	
	
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #_, 

		contour_list = []
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > 50:
				contour_list.append(contour)


		vertices = []
		for c in contour_list:
			vertices.append(find_vertices(c, h, w))
	
		img2 = cv2.imread(path+p)
	
		for i, v in enumerate(vertices):
			crop_img = img[v[0][1]:v[1][1], v[0][0]:v[1][0]] # img[y:y+h, x:x+w]
			shape = crop_img.shape
			#cv2.rectangle(img2, v[0], v[1], (255,0,0), 2)
			#print (v)
			crop_file = save_path + p[:-4] + "_" + str(i).zfill(4) + ".png"
			#print (p, crop_file)
			cv2.imwrite(crop_file, crop_img)
		
			boundary.append([crop_file, v])
		

		#cv2.imwrite(save_path + 'bounding_box_' + p, img2)
	

	df = pd.DataFrame(boundary, columns=['file_name','vertices'])
	print (df.shape)
	#df.to_csv('./boundary/boundary_{}.csv'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H %M %S')), sep='\t', index=False)



path = './data/test_rotation10/' #'/run/media/shalini/Seagate Expansion Drive/Shalini/Neyyvaasal Dataset/neyvaasal 3rd 3-11-17 pos and photos/100MSDCF/' #'./data/neyvaaasal 2nd 13-10-17 pos and photos/100MSDCF/' #landscape_set3/' #neyvaasal 1st 3-11-17 pos and photos/101MSDCF/' #test_rotation3/' #test_crop/' #'./data/landscape_set1/' #original_marker_images/'#landscape_set1/'
save_path = './data/test_rotation11/'

extract_marker(path, save_path)
