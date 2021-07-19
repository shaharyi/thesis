import time
import sys
import os
import copy
import multiprocessing
import warnings
import cPickle
import cProfile
import heapq
import operator
import logging
from pprint import pformat
from pdb import set_trace, pm
from math import log, sqrt, atan
from collections import OrderedDict, deque
from itertools import *
from logging import debug

import dicom
import pygame
import pygame.gfxdraw
from pygame.locals import *
from numpy import *
import numpy
from numpy.linalg import *
from scipy import interpolate, signal, ndimage, io, misc
from scipy.spatial import KDTree, distance
import scipy.linalg

import __main__

MATLAB_SHRINK_FACTOR = 2

def read_matlab_data(filename):
    data = io.loadmat(filename)['Cube']
    f = MATLAB_SHRINK_FACTOR
    data = data[::f,::f,::f]
    data = 255*data
    return data

def mark(data):
    total = float(sum(data>0))
    cum = 0
    for x in range(data.shape[0]):
        #if (any(data[x,:,:])>0):
        mass = sum(data[x,:,:]>0)
        cum += mass
        grade = 1-cum/total
##        debug('%d: %d %d %g', x, mass, cum, grade)
        debug('%g', grade)

if __name__=='__main__':
    if not os.path.isdir('Logs'):
        os.mkdir('Logs')       
    ts = time.strftime("%Y%m%d%H%M%S")
    logging.basicConfig(level=logging.DEBUG,
##                        format='%(asctime)s %(message)s',
                        format='%(message)s',
                        datefmt='%H:%M:%S',
                        filename='Logs/serial'+ts+'.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    #formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    #console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)    
    debug('* Started logging')
            
    data = read_matlab_data('../data_synth/Cube.mat')
    mark(data)
