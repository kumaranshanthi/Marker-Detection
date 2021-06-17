# Python program for Detection of a 
# specific color using OpenCV with Python
import cv2
import numpy as np 
import os
 
#filename = 'DSC01914_1.JPG' #'DSC00272_2.JPG'

def normalized(rgb):

        norm=np.zeros(rgb.shape,np.float32)
        norm_rgb=np.zeros(rgb.shape,np.uint8)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        sum=b+g+r+np.ones(rgb.shape[:-1])

        norm[:,:,0]=b/sum*255.0
        norm[:,:,1]=g/sum*255.0
        norm[:,:,2]=r/sum*255.0

        norm_rgb=cv2.convertScaleAbs(norm)
        return norm_rgb

def find_vertices(contour):
	
	min_x = 10000
	min_y = 10000
	max_x = 0
	max_y = 0
	
	gap = 5
	
	for c in contour:
		
		#print (c[0])
		x = c[0][0]
		y = c[0][1]
		
		#print (x)
		#print (y)
		
		if x < min_x:
			min_x = c[0][0]
			
		if y < min_y:
			min_y = c[0][1]
			
		if x > max_x:
			max_x = c[0][0]
			
		if y > max_y:
			max_y = c[0][1]
			
	return [(min_x-gap, min_y-gap), (max_x+gap, max_y+gap)]
	
def extract_vertices(contours):
	
	vertices = []
	
	for c in contours:
		print (c[0][0])
		vertices.append(c[0][0])

	return vertices	

path = './data/landscape_set1/' #original_marker_images/'#landscape_set1/'
save_path = './data/unclear_markers/'

for p in os.listdir(path):

	img = cv2.imread(path+p)

	#rgb_norm = normalized(frame)

	#print (rgb_norm)
	#hsv = frame
	#hsv = cv2.cvtColor(rgb_norm, cv2.COLOR_BGR2HSV)

	lower_red = np.array([170,170,170])
	upper_red = np.array([255,255,255])
	
	mask = cv2.inRange(img, lower_red, upper_red)
	
	#lower_red = np.array([0,0,0])
	#upper_red = np.array([15,255,255])
	
	#mask2 = cv2.inRange(hsv, lower_red, upper_red)

	#mask = mask1 + mask2
	
	res = cv2.bitwise_and(img,img, mask= mask)
	cv2.imwrite(save_path + 'hsv_' + p,res)	
	
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #_, 

	contour_list = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > 100 :
		    contour_list.append(contour)


	vertices = []
	for c in contour_list:
		#print (c)
		vertices.append(find_vertices(c))

	for v in vertices:
		cv2.rectangle(img, v[0], v[1], (255,0,0), 2)

	cv2.imwrite(save_path + 'bounding_box_' + p, img)

	
	"""

	"""
