import cv2
import numpy
import matplotlib.pyplot as plt


##defining the visibility function
def visibility(p,q,depth):
    x1,y1,z1 = p
    x2,y2,z2 = q
    for x in range(min(x1,x2)+1,max(x1,x2)):
        #calculate the corresponding y_coordinate.
        slope_y = (y2-y1)/(x2-x1)
        y = y1+slope_y*(x-x1)
        y = int(y)
        height = depth[x][y]

        #Calculate the corresponding z coordinate.
        slope_z = (z2-z1)/(x2-x1)
        z = z1+slope_z*(x-x1)
        #print(height,z)
        if(height > z):
            return False
    return True

# Creating the L-channel
img = cv2.imread('emma.jpg')
img_L = 0.212*img[:,:,2] + 0.715*img[:,:,1]+0.072*img[:,:,0]
img_L = img_L.astype('float32')

#Creating Base and depth layers
base = cv2.bilateralFilter(img_L,20,75,75)
detail = cv2.subtract(img_L.astype('float32'),base)

#Creating the depth map.
F_b = 0.8
F_d = 0.6

base_N = base/255
detail_N = detail/255
depth = F_b*base_N + F_d*detail_N

#Creating a point source of light
x_co = 10
y_co = 10
height = 1.2
p = (x_co,y_co, height)

#Creating new image based on visibility.
d_img = numpy.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        q = (i,j,depth[i][j])
        if(visibility(p,q,depth)):
            d_img[i][j] = img[i][j]

#python uses bgr matplotlib rgb so changing colour space.
b = d_img[:,:,0]
g = d_img[:,:,1]
r = d_img[:,:,2]
d_img_rgb = numpy.zeros(img.shape)
d_img_rgb[:,:,0] = r
d_img_rgb[:,:,1] = g
d_img_rgb[:,:,2] = b

plt.imshow(d_img_rgb.astype(int))
plt.show()


        
        
        
