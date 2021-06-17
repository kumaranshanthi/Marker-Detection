#from PIL import Image
import numpy as np
import cv2
import imutils
import os
import random

path = './data/test_rotation6/'#unclear_markers/'
save_path = './data/test_rotation10/'

for file_name in os.listdir(path):

    img = cv2.imread(path+file_name)
    print (path+file_name)
    
    n = random.randint(0,180)
    
    for angle in n + np.arange(0, 180, 30):
	    rotated = imutils.rotate_bound(img, angle)
	    cv2.imwrite(save_path + "{}_{}.jpg".format(file_name[:-4], angle), rotated)
    
    #im3 = img.rotate((width, height), Image.ANTIALIAS)
    #im3.save(save_path + file_path)
