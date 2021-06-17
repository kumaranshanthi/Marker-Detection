from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

filename = 'DSC00273_2.JPG'
#'./data/unclear_markers/DSC01299_1.JPG' #'DSC01914_1.JPG' #'./data/neg_obj/DSC01217_1.JPG' #'cropped.jpg' #'./data/neg_obj/DSC01212_2.JPG' #DSC01256_5.JPG' #'down_IMG_20180111_001757216.jpg' #'temp.jpg' #'DSC00272_2.JPG'

img = cv2.imread(filename)
height, width, dim = img.shape

#img = img[(height/4):(3*height/4), (width/4):(3*width/4), :]
#height, width, dim = img.shape

#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_vec = np.reshape(img, [height * width, dim] )

kmeans = KMeans(n_clusters=3)
kmeans.fit( img_vec )

unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
sort_ix = np.argsort(counts_l)
sort_ix = sort_ix[::-1]

fig = plt.figure()
ax = fig.add_subplot(111)
x_from = 0.05

for cluster_center in kmeans.cluster_centers_[sort_ix]:
    print (cluster_center)
    ax.add_patch(patches.Rectangle( (x_from, 0.05), 0.29, 0.9, alpha=None,
                                    facecolor='#%02x%02x%02x' % (cluster_center[2], cluster_center[1], cluster_center[0] ) ) )
    x_from = x_from + 0.31

plt.show()
