#*******************************************************
#* Copyright (c) 2020 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++  Artelys Knitro 12.2 Python API
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


'''NumPy support in Python interface for Artelys Knitro.

This module activates the support of NumPy arrays in the
Python interface for Artelys Knitro.

By adding these methods, Knitro callbacks are able to work
directly with NumPy arrays, instead of Python's lists.

This source is provided for informational purposes.
'''

import ctypes
import numpy as np
from ..wrapper import *
from ..ktr import *

#-------------------------------------------------------------------------------
#     METHODS AND CLASSES TO WORK WITH C ARRAYS AND POINTERS
#     SPECIALIZATION FOR NUMPY ARRAYS
#-------------------------------------------------------------------------------

@staticmethod
def _cIntArray (npArray):
    '''Build a 'ctypes' array of integers from a NumPy array.
    '''
    if npArray is not None:
        n = len (npArray)
        if n > 0:
            if type(npArray) == np.ndarray and npArray.dtype == np.int32:
                return npArray.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            else: # in case of a non NumPy array or non 32-bit integer element type, simply create a new array
                return (ctypes.c_int * n) (*npArray)
    return None

@staticmethod
def _cDoubleArray (npArray):
    '''Build a 'ctypes' array of doubles from a NumPy array.
    '''
    if npArray is not None:
        n = len (npArray)
        if n > 0:
            if type(npArray) == np.ndarray and npArray.dtype == np.float:
                return npArray.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            else:  # in case of a non NumPy array or non double element type, simply create a new array
                return (ctypes.c_double * n) (*npArray)
    return None

@staticmethod
def _userArray (size, cArray):
    '''Build a NumPy array from a 'ctypes' array.
    '''
    if cArray:
        return np.ctypeslib.as_array (cArray, tuple ([size]))
    else:
        return None

@staticmethod
def _userToCArray (npArray, cArray):
    '''Copy the content of a NumPy array to a 'ctypes' array.
    '''
    if npArray is not None:
        if npArray.ctypes.data != ctypes.addressof(cArray.contents):
            for i in xrange (len (npArray)):
                cArray[i] = npArray[i]

@staticmethod
def _cToUserArray (size, cArray, npArray):
    '''Copy the content a 'ctypes' array to a NumPy array.
    '''
    if cArray:
        if type(npArray) == np.ndarray:
            npArray = np.ctypeslib.as_array (cArray, tuple ([size]))
        else:
            npArray[:] = cArray

# Overwrite methods of KN_array_handler.
KN_array_handler._cIntArray = _cIntArray
KN_array_handler._cDoubleArray = _cDoubleArray
KN_array_handler._userArray = _userArray
KN_array_handler._userToCArray = _userToCArray
KN_array_handler._cToUserArray = _cToUserArray

# For retrocompatibility purposes
KTR_array_handler._cIntArray    = _cIntArray
KTR_array_handler._cDoubleArray = _cDoubleArray
KTR_array_handler._userArray    = _userArray
KTR_array_handler._userToCArray = _userToCArray
KTR_array_handler._cToUserArray = _cToUserArray
