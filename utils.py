# -*- coding: utf-8 -*-
import os, sys
import traceback
import numpy as np
import time
import numba as nb


@nb.jit(nopython=True)
def bitwise(img: np.ndarray, mask: np.ndarray) -> np.ndarray:

    res = None
    h, w = 0, 0

    # Check whether the image is Gray or RGB to do diffferent processes
    if img.ndim == 3:
        h, w, c = img.shape
        res = np.zeros((h, w, c))

    else:
        h, w = img.shape
        res = np.zeros((h, w))
        
    # For numba optimization, iteration is faster than numpy vectorize
    # loop the mask to check which pixels should get through
    for n in range(h):
        for m in range(w):
            if mask[n,m] == 1:
                res[n,m] = img[n,m]

    return res


@nb.jit(nopython=True)
def get_Euclideandist(point1: np.ndarray, point2:np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(point1 - point2, 2)))
    

class Timer:

    def __init__(self) -> None:
        self.start = time.time()

    def time(self) -> float:
        '''
        Return the period time but doesn't reset the start time
        '''
        return time.time() - self.start

    def timeslice(self) -> float:
        '''
        1. Ruturn the period time
        2. Reset current time as the start time
        '''
        end = time.time()
        t = end - self.start
        self.start = end
        return t

'''
Error
'''
def get_error_info(e:Exception):

    exc_type, exc_obj, exc_tb = sys.exc_info() # get Call Stack
    filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1] # get filename
    filename, lineno, function_name = traceback.extract_tb(exc_tb)[-1][0:3]

    return "File: {}, line {}, in {} - [{}] {}".format(filename, lineno, function_name, e.__class__.__name__, str(e))
'''
Error
'''
