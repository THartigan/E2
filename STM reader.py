from NSFopen.read import read

#import os
#from os import listdir
#from os.path import isfile, join

import numpy as np
import pandas
import h5py
import math
import collections

#import cv2 as cv
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.axes_grid1.colorbar import colorbar

#from NSFopen.read import nid_read
#r'C:\Users\Max Schubert\Documents\work\STM images\3;1\(file name).nid path
afm = read(r'C:\Users\Max Schubert\Documents\work\STM images\NidFilesScans\Image13910.nid')
data = afm.data
param = afm.param
print(data.keys())
print(param.keys())


forward_current = data['Image']['Forward']['Tip Current']
#AV = np.average(forward_current)
#forward_current = forward_current - AV
transform = np.fft.fft2(forward_current)
transform[0] = 0
l = len(transform) - 1
while l != -1:
    transform[l][0] = 0
    l = l - 1
#print(transform)
transform = np.fft.fftshift(transform)

l = len(transform) - 2
L = len(transform) - 2
arr = [[0, 0, 0] for i in range(7)]

while L > 0:
    while l > 0:
        if transform[L][l] > transform[L + 1][l + 1]:
            if transform[L][l] > transform[L + 1][l - 1]:
                if transform[L][l] > transform[L - 1][l + 1]:
                    if transform[L][l] > transform[L - 1][l - 1]:
                        if transform[L][l] > arr[6][0]:
                            arr[6][0] = transform[L][l]
                            arr[6][1] = l
                            arr[6][2] = L
                            a = 6
                            while a > 1:
                                if arr[a][0] > arr[a - 1][0]:
                                    b = arr[a - 1][0]
                                    c = arr[a - 1][1]
                                    d = arr[a - 1][2]
                                    arr[a - 1][0] = arr[a][0]
                                    arr[a - 1][1] = arr[a][1]
                                    arr[a - 1][2] = arr[a][2]
                                    arr[a][0] = b
                                    arr[a][1] = c
                                    arr[a][2] = d
                                a = a - 1
                            
                                
        if transform[L][l] == 0:
            if transform[L + 1][l] == 0:
                if transform[L][l + 1] == 0:
                    if transform[L - 1][l] == 0:
                        if transform[L][l - 1] == 0:
                            arr[0][1] = l
                            arr[0][2] = L
        l = l - 1
    l = len(transform) - 2
    L = L - 1

l = len(transform) - 1
L = len(transform) - 1
while L >= 0:
    while l >= 0:
        transform[L][l] = 0
        l = l - 1
    l = len(transform) - 1
    L = L - 1
a = 6
while a > 0:
    b = arr[a][1]
    c = arr[a][2]
    d = arr[a][0]
    transform[b][c] = 10 ** -8
    a = a - 1
forward_current = np.fft.ifft2(transform)
forward_current = np.fft.fftshift(forward_current)
Distances = [((((arr[i + 1][1] - arr[0][1]) ** 2) + ((arr[i + 1][2] - arr[0][2]) ** 2)) ** 0.5) for i in range(6)]
Distances = sorted(Distances)
print(Distances)
#distances = [1000000, 1000000, 1000000]


#print(arr)
#plt.imshow(abs((transform) * (10 ** 8)), cmap = "Blues")
plt.imshow((abs((forward_current) * 2.5 * (10 ** 11)) ** 1), cmap = "Blues")
plt.show()
