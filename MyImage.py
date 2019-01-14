#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import math
import queue as q
import collections
import enum

# In[7]:

class Direction(enum.Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class Threshold(enum.Enum):
    OTSU = 1
    BINARY = 2
    ADAPTIVE_GAUSSIAN = 3
    ADAPTIVE_MEAN = 4

class MyImage:
    eps = 3
    BLACK = 0
    WHITE = 255
    MIN_CLUSTER_PIXEL_NUM = 6
    
    def __init__(self, img_path, box_size = 2, box_pixel_leeway = 2, center_box_size = 1):
        self.img = cv2.imread(img_path)
        self.cluster_img = np.zeros((self.img.shape[0], self.img.shape[1])) # ...what
        self.threshed = np.zeros((self.img.shape[0], self.img.shape[1]))
        self.num_clusters = 0
        self.centers = q.Queue()
        self.pixels_to_proc = q.Queue() # pixels in current cluster to process (?)
        self.box_size = box_size
        self.box_pixel_leeway = box_pixel_leeway
        self.min_cluster_pixel_num = (self.box_size+1)**2 - self.box_pixel_leeway
        assert self.min_cluster_pixel_num > 0
        self.center_box_size = center_box_size
        #self.curr_cluster = 0

        self.threshold(Threshold.ADAPTIVE_GAUSSIAN)
        return
    
    def add_pixel_to_cluster(self, x_curr, y_curr, x, y):
        self.cluster_img[x_curr,y_curr] = self.cluster_img[x,y]
        return
    
    def add_pixel_to_next_cluster(self, x,y):
        #self.curr_cluster += 1
        self.num_clusters += 1
        self.cluster_img[x,y] = self.num_clusters
        return
    
    def pixels_are_similar(self, x1,y1,x2,y2, eps):
        pix1, pix2  = self.img[x1,y1], self.img[x2,y2]
        dist = math.sqrt(np.sum((pix1-pix2)**2))
        return dist <= eps
    
    def threshold(self, method, threshold_val = 70):
        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        if(method == Threshold.OTSU):
            ret, self.threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif(method == Threshold.BINARY):
            ret, self.threshed = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        elif(method == Threshold.ADAPTIVE_GAUSSIAN):
            self.threshed = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                                 cv.THRESH_BINARY,11,2)
        elif(method == Threshold.ADAPTIVE_MEAN):
            self.threshed = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                                                 cv.THRESH_BINARY,11,2)
        assert(self.threshed.shape == self.img.shape)
        return

    """
    This function becomes unnecessary if we decide to go with the new route, but let's 
    just leave it here for now.
    """
    def is_in_cluster(self, x,y,as_x, as_y):
        assert as_x >=0 and as_x <= self.box_size and as_y >= 0 and as_y <= self.box_size
        
        this_color = self.img[x,y]
        clsts_num_dict = collections.Counter()
        x_init, y_init = x - as_x, y - as_y
        x_end, y_end = x_init + self.box_size, y_init + self.box_size
        for i in range(x_init, x_end + 1):
            for j in range(y_init, y_end + 1):
                if(i < 0 or j < 0 or i >= self.img.shape[0] or j >= self.img.shape[1]):
                    continue
                if(not self.img[i,j] == this_color or self.cluster_img[i,j] == 0):
                    continue
                if(self.cluster_img[i,j] in clsts_num_dict):
                    clsts_num_dict[self.cluster_img[i,j]] += 1
                else:                    
                    clsts_num_dict[self.cluster_img[i,j]] = 1
                    
        if(clsts_num_dict.most_common(1)[0][1] >= self.min_cluster_pixel_num):            
            return True, clsts_num_dict.most_common(1)[0]
        
        return False, (-1,-1)
    
    def is_center(self, x,y):
        this_color = self.threshed[x,y]
        for i in range(-self.center_box_size, self.center_box_size + 1):
            for j in range(-self.center_box_size, self.center_box_size + 1):
                if(x+i < 0 or x+i >= self.img.shape[0] or y+j < 0 or y+j >= self.img.shape[1]):
                    return False
                if(not self.threshed[x + i, y + j] == this_color):
                    return False
        return True

    """
    Given pixel (x, y)
    Find a cluster center. Add every pixel touching it to the cluster, add them to the queue.
    Now, go to a newly clustered pixel, and look at everything touching this pixel. 
    If all of the same color, and are either unclustered or of the same cluster as current cluster, 
    then cluster all those, and add them to the queue to be "processed"
    Else, don't add any to the queue, don't cluster those. Continue with next pixel in queue

    So, given the pixel, must cluster around this pixel

    Eventual TODOs: maybe decrease the strictness of condition for adding pixels to the current 
    cluster, but this is more polish than requirement for now. 
    """

    # TODO: Test function
    def cluster_around_this_pixel(self, x, y):
        assert x >=0 and x < self.img.shape[0] and y >= 0 and y < self.img.shape[1]

        # TODO: Test indexing
        # TOOD: Look into including diagonal neighbors
        def get_neighbor_pos(dir, x, y):
            assert dir in Direction
            if dir == Direction.UP:
                if y > 0:
                    return x, y-1
                return -1, -1
            elif dir == Direction.RIGHT:
                if x < self.box_size:
                    return x+1, y
                return -1, -1
            elif dir == Direction.DOWN:
                if y < self.box_size:
                    return x, y+1
                return -1, -1
            else:
                if x > 0:
                    return x-1, y
                return -1, -1

        pixel_color = self.thresehd[x,y]
        for dir in Direction:
            new_x, new_y = get_neighbor_pos(dir, x, y)
            if new_x == new_y == -1:
                continue
            # TODO: Look into alternative conditions for deciding to cluster
            elif self.cluster_img[new_x,new_y] == 0 and self.threshed[new_x,new_y] == pixel_color:
                self.add_pixel_to_cluster(new_x, new_y, x, y)
                self.pixels_to_proc.put((new_x, new_y))
        return



    """
    Function that gives every pixel a cluster number. This is done by
    changing the value of pixels in cluster_img
    For eg, if pixel (x,y) were in cluster 5, then cluster_img[x,y] = 5
    0 - unclustered. Every other number stands for a cluster number.
    
    Perform this by implementing DBSCAN
    """
    
 
    def get_clusters(self):
        for x in range(0, self.img.shape[0]):
            for y in range(0, self.img.shape[1]):
                if self.cluster_img[x][y] == 0 and self.is_center(x, y): 
                    # if is a center and hasn't been clustered at all yet, make new cluster and get its cluster!
                    self.add_pixel_to_next_cluster(x, y)
                    # add all pixels around to current cluster

                    for x_surr in range(x - 1, x + 2):
                        for y_surr in range(y - 1, y + 2):
                            if x_surr = x and y_surr = y:
                                continue # can skip analyzing itself
                            if x_surr < 0 or x_surr >= self.img.shape[0] or y_surr < 0 or y_surr >= self.img.shape[1]:
                                continue
                            self.add_pixel_to_cluster(x_surr, y_surr, x, y)
                            coord = (x_surr, y_surr)
                            self.pixels_to_proc.put(coord)
                    while not self.pixels_to_proc.empty():
                        coord = self.pixels_to_proc.get()
                        cluster_around_this_pixel(coord[0], coord[1])
                    
                else:
                    # it's noise
                    continue
        
        
        

# In[9]:


x = MyImage("cell_img2.png")
#plt.imshow(x.img_)

#plt.imshow(x.img)
x.img.shape
#x.img[659:669, :10, :]


# In[10]:

"""
plt.imshow(x.img[100:150, 200:250], cmap = "gray")


# In[15]:


gray = cv2.cvtColor(x.img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap = 'gray')


# In[29]:


area = int(5*gray.shape[0]*gray.shape[1]/(400*400))
box = 2*area + 1;
print(area)
print(box)
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY,15,5)
plt.imshow(th3, cmap = 'gray')
ret, thresh = cv2.threshold(th3,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# In[21]:


plt.imshow(thresh, cmap = 'gray')


# In[30]:


denoised= cv2.fastNlMeansDenoising(thresh)
plt.imshow(denoised, cmap = 'gray')


# In[22]:


## noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


# In[23]:


plt.imshow(sure_fg, cmap = "gray")


# In[24]:


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0


# In[17]:


markers = cv2.watershed(x.img,markers)
img = x.img
img[markers == -1] = [255,0,0]
plt.imshow(img)


# In[25]:


_, markers = cv2.connectedComponents(thresh)


# In[26]:


plt.imshow(markers)


# In[27]:


print(_)


# In[ ]:




"""
