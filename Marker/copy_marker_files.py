import numpy as np
import pandas as pd
import cv2

df = pd.read_csv('marker_set3.txt', sep='\n')
col0 = 'DSC0' + df[df.columns[0]].astype(str) + '.JPG'
l = col0.astype(list)
path = '/run/media/shalini/Seagate Expansion Drive/Shalini/Neyyvaasal Dataset/neyvaasal 3rd 3-11-17 pos and photos/100MSDCF/' #'./data/neyvaaasal 2nd 13-10-17 pos and photos/100MSDCF/'
save_path = './data/landscape_set5/'

for f in l:
	print (path+f)
	img = cv2.imread(path+f)
	
	#if img.shape != None:
	cv2.imwrite(save_path+f, img)
