import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = 'DSC00272_2.JPG' #'./data/unclear_markers/DSC01299_1.JPG' #'DSC00272_2.JPG' #'down_IMG_20180111_001757216.jpg' #'DSC00272_2.JPG'#'DSC00271_2.JPG'

image = cv2.imread(filename)
color = ('b','g','r')
for i,col in enumerate(color):
	histr = cv2.calcHist([image],[i],None,[256],[0,256])
	plt.plot(histr,color = col)
	plt.xlim([0,256])
#plt.show()

# define the list of boundaries
boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

"""
# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
 
	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)
"""
(lower, upper) = ([25, 146, 190], [62, 174, 250])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (170, 0, 0), (180, 255, 255))
mask2 = cv2.inRange(hsv, (0, 0, 0), (10, 255, 255))

mask1 = mask1.astype('bool')
mask2 = mask2.astype('bool')

mask_full = mask1 + mask2

lower = np.array([180,180,150])
upper = np.array([255,255,255])

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")
 
# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)
cv2.imwrite("mask1.png",mask_full)
# show the images
#cv2.imshow("images", hsv)#output)
#cv2.waitKey(0)

