#IMAGE_FILENAME      = '../data/Head1/MR000057.dcm'
#DATA                = '../mri_data'
DATA                = '../SE000000'
IMG_EXT             = '.tif' 
#DATA                = '../mrbrain-8bit'
#IMAGE               = 'mrbrain-8bit%03i.tif'
#DATA                = '../ct_data'
#IMAGE               = 'CThead.%i'
#DATA                = '../data'
#IMAGE               = 'MR0000%02i.dcm'
IMAGE_FILENAME      = '../mri_data/MRbrain.1'
IMG_SIZE            = (256,256)
WHITE               = (255,255,255)

import time
import sys
import os
import copy
import multiprocessing
import warnings
import cPickle
from pprint import pprint
from pdb import set_trace,pm
from math import log, sqrt, atan
from bisect import bisect
from itertools import count, izip
from collections import OrderedDict

import dicom
import PIL
import pygame
import pygame.gfxdraw
from pygame.locals import *
import numpy as np
from scipy import interpolate, signal, ndimage 

def wait_event(event=KEYDOWN,show=None):
    pygame.event.clear()
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT or \
            (e.type == KEYDOWN and e.key == K_ESCAPE):
            pygame.quit()
            exit()
        if e.type == event:
            break

def histeq(im,nbr_bins=256):
   #get image histogram
   imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize
   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(im.flatten(),bins[:-1],cdf)
   return im2.reshape(im.shape).astype('uint32'), cdf

def show_slice(surf, a):
    ##a=histeq(a)[0]
    #a *= 255.0 / a.max()
    #a=a.astype('uint32')
    #a += (a<<8) + (a<<16) + (255<<24)
    ##a = a.astype('uint32')
    ##a += 256*a + 256*256*a + 256*256*256*255
    #s = a.tostring()
    #img = pygame.image.frombuffer(s,a.shape,'RGBA')
    ##img = pygame.transform.smoothscale(img, IMG_SIZE)
    ##img = img.convert(8)
    ##img = PIL.Image.fromarray(np.uint8(a))
    #screen.blit(img,(0,0))
    if a.shape != surf.get_size():
        surf = pygame.display.set_mode(a.shape)
    pygame.surfarray.blit_array(surf, a)
    pygame.display.flip()
    
def get_scribble(surf):
    quit = False
    while not quit:
        #for event in pygame.event.get():
        e = pygame.event.wait()
        if e.type == MOUSEMOTION:
            if e.buttons[0]:
                x2, y2 = pygame.mouse.get_pos()
                pygame.gfxdraw.line(surf, x1,y1,x2,y2, WHITE)
                x1,y1 = x2,y2
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 1:
                x1,y1 = e.pos
        elif e.type == QUIT or \
            (e.type == KEYDOWN and e.key == K_ESCAPE):
            quit = True
        pygame.display.flip()

def expand(data):
    "expand Z by 2 via interp"
    M=np.array([[.5,0,0],[0,1,0],[0,0,1]])
    bb=ndimage.affine_transform(data,M,prefilter=False,order=3,
                                output_shape=(2*data.shape[0],)+data.shape[1:])
    newshape = (2*data.shape[0],) + data.shape[1:]
    bb.reshape(newshape)
    return bb

def read_data_dicom():
    "return data"
    dirlist = sorted(os.listdir(DATA))
    paths = [os.path.join(DATA,p) for p in dirlist]
    aa = [dicom.ReadFile(p).pixel_array for p in paths]

    #sometimes the last image has different size, so drop bad imgs
    the_shape = aa[0].shape
    aa = np.array([a for a in aa if a.shape==the_shape])

    ##aa = array([histeq(a) for a in aa])
    amin,amax = aa.min(), aa.max()
    aa = (aa - amin) * 255.0 / (amax-amin)
    aa = aa.round().astype('uint8')
    a = aa.round().astype('uint32')
    a += (a<<8) + (a<<16) + (255<<24)
    print('Read {} images'.format(len(aa)))
    #a=histeq(a)[0]
    #a *= 255.0 / a.max()
    ##a=a.astype('ubyte')
    #a=a.astype('uint32')
    #a += (a<<8) + (a<<16) + (255<<24)
    ##a += 256*a + 256*256*a + 256*256*256*255
    #aa = expand(aa)
    return a

def read_data(r=256,c=256):
    dirlist = sorted(os.listdir(DATA),key=lambda x: int(x.split('.')[1]))
    half=len(dirlist)/2
    paths = [os.path.join(DATA,p) for p in dirlist[half-5:half+5]]
    nimages = len(paths)
    a = np.array([np.fromfile(p,dtype='>u2') for p in paths])
    #a = histeq(a)[0]

    aa=a.reshape(nimages,r,c)
    bb=expand(aa)
    nimages = 2*nimages
    bb = np.array([(x - x.min()) * 255.0 / (x.max()-x.min()) for x in bb])
##    b=np.zeros((nimages*2-1,)+a.shape[1:], dtype=a.dtype)
##    for i in range(0,2*nimages-2,2):
##        b[i] = a[i/2]
##        b[i+1]=(a[i/2] + a[i/2+1])/2
##        print(i)
####    set_trace()
##    b[2*nimages-2] = a[nimages-1]
    a = bb.round().astype('uint32')
    a += (a<<8) + (a<<16) + (255<<24)
    print('Read {} images'.format(nimages))
    ##a.shape = (nimages, r, c)
    #a = a.T
    return a

def read_data_DICOM():
    import dicom
    if filename.endswith('.dcm'):
        dcm = dicom.ReadFile(filename)
        a = dcm.pixel_array
        
def read_data_PIL():
    listdir = sorted(os.listdir(DATA))
    paths = [os.path.join(DATA,p) for p in listdir if p.endswith(IMG_EXT)]
    nimages = len(paths)
    print('Read {} images'.format(nimages))
    I = PIL.Image.open(paths[0]).getdata()
    data = np.array([PIL.Image.open(p,'L').getdata()  for p in paths])
    #data = np.array([pygame.image.load(p).get_buffer() for p in paths])
    data.shape = (nimages,) + I.size
    #data = data.T
    return data
                
if __name__=='__main__':
    global data
    data = read_data()
    #raise SystemExit
    screen = pygame.display.set_mode(IMG_SIZE)    
    for i in range(0,data.shape[1]):
        show_slice(screen, data[:,i,:])
        print(i)
        wait_event()
    scribble = get_scribble(screen)
    slice = data[:,:,120]
    show_slice(screen, slice)
    scribble = get_scribble(screen)
    #wait_event()
    pygame.quit()
