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


# In[7]:


class MyImage:
    eps = 3
    THRESHOLD_OTSU = 1
    THRESHOLD_BINARY = 2
    THRESHOLD_ADAPTIVE_GAUSSIAN = 3
    BLACK = 0
    WHITE = 255
    MIN_CLUSTER_PIXEL_NUM = 6
    
    def __init__(self, img_path, box_size = 2, box_pixel_leeway = 2, center_box_size = 1):
        self.img = cv2.imread(img_path)
        self.cluster_img = np.zeros((self.img.shape[0], self.img.shape[1]))
        self.num_clusters = 0
        self.centers = q.Queue()
        self.pixels_to_proc = q.Queue()
        self.box_size = box_size
        self.box_pixel_leeway = box_pixel_leeway
        self.min_cluster_pixel_num = (self.box_size+1)**2 - self.box_pixel_leeway
        assert self.min_cluster_pixel_num > 0
        self.center_box_size = center_box_size
        #self.curr_cluster = 0
        return
    
    def add_pixel_to_cluster(x_curr, y_curr, x, y):
        self.cluster_img[x_curr,y_curr] = self.cluster_img[x,y]
        return
    
    def add_pixel_to_cluster(x,y):
        #self.curr_cluster += 1
        self.num_clusters += 1
        self.cluster_img[x,y] = self.num_clusters
        return
    
    def pixels_are_similar(x1,y1,x2,y2, eps):
        pix1, pix2  = self.img[x1,y1], self.img[x2,y2]
        dist = math.sqrt(np.sum((pix1-pix2)**2))
        return dist <= eps
    
    def threshold(method, threshold_val = 0):
        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        if(method == THRESHOLD_OTSU):
            ret, self.threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif(method == THRESHOLD_BINARY):
            ret, self.threshed = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        else:
            ret, self.threshed = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        
    def is_in_cluster(x,y,as_x, as_y):
        assert as_x >=0 and as_x <= self.box_size and as_y >= 0 and as_y <= self.box_size
        
        this_color = self.img[x,y]
        clsts_num_dict = collections.Counter()
        x_init, y_init = x - x_as, y - y_as
        x_end = x_init + self.box_size, y_init + self.box_size
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
    
    def is_center(x,y):
        this_color = self.img[x,y]
        for i in range(-self.center_box_size, self.center_box_size + 1):
            for j in range(-self.center_box_size, self.center_box_size + 1):
                if(x+i < 0 or x+i >= self.img.shape[0] or y+j < 0 or y+j >= self.img.shape[1]):
                    return False
                if(not self.img[x + i, y + j] == this_color):
                    return False
        return True
  
    def get_clusters():
        #num_centers = get_centers()
        while(not centers.empty()):
            x,y,cluster_number = centers.get()
            for i in [-3,0,3]:
                for j in [-3,0,3]:
                    if(i < 0 or j < 0 or i >= self.img.shape[0] or j >= self.img.shape[1]):
                        continue
                    if(self.img[x + i, y + j] == BLACK and i > -3 and i < 3 and j > -3 and j < 3):
                        self.cluster_img[x + i, y + j] = cluster_number
                    elif(self.img[x + i, y + j] == BLACK):
                        pixels_to_proc.put([x+i, y+j])
                    else:
                        self.cluster_img[x + i, y + j] = 1
                    
        return
    """    
      def get_centers():
        x_min, y_min = 2,2
        x_max, y_max = self.img.shape - 2
        num_centers = 0
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                if(is_center(i,j)):
                    self.centers.put([i,j, num_centers + 2])
                    num_centers += 1
        return num_centers
    """  


# In[9]:


x = MyImage("cell_img2.png")
#plt.imshow(x.img_)

#plt.imshow(x.img)
x.img.shape
#x.img[659:669, :10, :]


# In[10]:


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




