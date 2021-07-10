#*******************************************************
#* Copyright (c) 2020 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++  Artelys Knitro 12.2 Python API
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


'''Python interface of Artelys Knitro providing a wrapper for
Artelys Knitro C API.

Multiple instances of the Knitro context can be constructed and run
simultaneously in separate threads. Each instance of Knitro context
allocates a distinct Knitro instance in C.

Applications should create a Knitro context and call its
methods, similar to Python code in examples/Python.

This source is provided for informational purposes.
'''

import sys
import itertools
import ctypes, ctypes.util
# Safe check for numpy/list
from collections.abc import Iterable

from .constants import *


#-------------------------------------------------------------------------------
#     Python 2 / Python 3 compatibility handling
#-------------------------------------------------------------------------------

try:
    xrange
except NameError:
    xrange = range

#-------------------------------------------------------------------------------
#     STATIC LOAD METHOD FOR THE NATIVE LIBRARY
#-------------------------------------------------------------------------------
try:
    _knitroLibraryFile = ctypes.util.find_library("knitro")
    if sys.platform == 'win32': # Windows
        if _knitroLibraryFile is None:
            _knitroLibraryFile = "knitro.dll"
        _knitro = ctypes.windll.LoadLibrary (_knitroLibraryFile)
    elif sys.platform == 'darwin': # Mac OS X
        if _knitroLibraryFile is None:
            _knitroLibraryFile = "libknitro.dylib"
        _knitro = ctypes.cdll.LoadLibrary (_knitroLibraryFile)
    else: # Linux, Solaris
        if _knitroLibraryFile is None:
            _knitroLibraryFile = "libknitro.so"
        _knitro = ctypes.cdll.LoadLibrary (_knitroLibraryFile)
except OSError as e:
    print ("Failed to load the Artelys Knitro native library '" + _knitroLibraryFile + "'.")
    print ("Make sure that your environment variables are set properly and Knitro/Python architectures match (32-bit/64-bit).")
    from unittest.mock import Mock
    _knitro = Mock()


#-------------------------------------------------------------------------------
#     SELECT APPROPRIATE API TYPE FOR KNLONG
#-------------------------------------------------------------------------------

KNLONG = ctypes.c_longlong
if sys.platform == 'win32' and ctypes.sizeof(ctypes.c_void_p) == 4: # Windows 32-bit
    KNLONG = ctypes.c_int


#-------------------------------------------------------------------------------
#     UTIL FUNCTIONS
#-------------------------------------------------------------------------------

def _checkRaise (fname, ret):
    if ret:
        raise RuntimeError ("Knitro-Python error: Return code for C function " + fname + "() is " + str(ret))
    return ret

def isiterable(obj):
    return isinstance(obj, Iterable)


#-------------------------------------------------------------------------------
#     FUNCTIONS AND CLASSES TO WORK WITH C ARRAYS ANDctypes.POINTERS
#-------------------------------------------------------------------------------

class KN_array_handler:
    '''Handler class for user-provided and C arrays.
    '''

    @staticmethod
    def _cIntPointer (x):
        '''Construct a 'ctypes'ctypes.POINTER to an integer from a Python integer.
        '''
        return ctypes.byref (ctypes.c_int (x))

    @staticmethod
    def _cDoublePointer (x):
        '''Construct a 'ctypes'ctypes.POINTER to a double from a Python float.
        '''
        return ctypes.byref (ctypes.c_double (x))

    @staticmethod
    def _cIntArray (pyArray):
        '''Construct a 'ctypes' array of integers from a Python list.
        '''
        if pyArray is not None:
            n = len (pyArray)
            if n > 0:
                return (ctypes.c_int * n) (*pyArray)
        return None

    @staticmethod
    def _cIntArray (pyArray):
        '''Construct a 'ctypes' array of integers from a Python list.
        '''
        if pyArray is not None:
            n = len (pyArray)
            if n > 0:
                return (ctypes.c_int * n) (*pyArray)
        return None

    @staticmethod
    def _cDoubleArray (pyArray):
        '''Construct a 'ctypes' array of doubles from a Python list.
        '''
        if pyArray is not None:
            n = len (pyArray)
            if n > 0:
                return (ctypes.c_double * n) (*pyArray)
        return None

    @staticmethod
    def _cStringArray (pyArray):
        '''Construct a 'ctypes' array of strings (char*) from a Python list.
        '''
        if pyArray is not None:
            n = len (pyArray)
            if n > 0:
                pyArrayCopy = map (lambda x: x.encode ('UTF-8'),  pyArray)
                return (ctypes.c_char_p * n) (*pyArrayCopy)
        return None

    @staticmethod
    def _userArray (size, cArray):
        '''Construct a Python list from a 'ctypes' array.
        '''
        if cArray:
            return [cArray[i] for i in xrange (size)]
        else:
            return None

    @staticmethod
    def _userToCArray (pyArray, cArray):
        '''Copy the content of a Python list to a 'ctypes' array.
        '''
        if pyArray is not None:
            for i in xrange (len (pyArray)):
                cArray[i] = pyArray[i]

    @staticmethod
    def _cToUserArray (size, cArray, pyArray):
        '''Copy the content a 'ctypes' array to a Python list.
        '''
        if cArray:
            pyArray[:] = cArray


def _cIntPointer (x):
    return KN_array_handler._cIntPointer (x)

def _cDoublePointer (x):
    return KN_array_handler._cDoublePointer (x)

def _cIntArray (userArray):
    return KN_array_handler._cIntArray (userArray)

def _cDoubleArray (userArray):
    return KN_array_handler._cDoubleArray (userArray)

def _cStringArray (userArray):
    return KN_array_handler._cStringArray (userArray)

def _userArray (size, cArray):
    return KN_array_handler._userArray (size, cArray)

def _userToCArray (userArray, cArray):
    KN_array_handler._userToCArray (userArray, cArray)

def _cToUserArray (size, cArray, userArray):
    KN_array_handler._cToUserArray (size, cArray, userArray)


#-------------------------------------------------------------------------------
#     DEFINITIONS OF KNITRO INTERNAL ctypes.Structures
#-------------------------------------------------------------------------------

class KN_context (ctypes.Structure):
    '''Wrapper for KN_context ctypes.Structure
    '''
    _fields_ = []
    def __str__(self):
        return "KN_context"

KN_context_ptr = ctypes.POINTER (KN_context)
KN_context_ptr.__str__ = lambda self: "KN_context_ptr (" + repr (self.contents) + ")"
KN_context_ptr.__hash__ = lambda self: str (self.contents).__hash__()
KN_context_ptr.__enter__ = lambda self: self
KN_context_ptr.__exit__ = lambda self, ex_val, ex_typ, ex_tb: KN_free(self)

class LM_context (ctypes.Structure):
    '''Wrapper for LM_context ctypes.Structure
    '''
    _fields_ = []
    def __str__(self):
      return "LM_context"
LM_context_ptr = ctypes.POINTER (LM_context)
LM_context_ptr.__str__ = lambda self: "LM_context_ptr (" + repr (self.contents) + ")"

class CB_context (ctypes.Structure):
    '''Wrapper for CB_context ctypes.Structure
    '''
    _fields_ = []
    def __str__(self):
      return "CB_context"
CB_context_ptr = ctypes.POINTER (CB_context)
CB_context_ptr.__str__ = lambda self: "CB_context_ptr (" + repr (self.contents) + ")"

class _KN_eval_request (ctypes.Structure):
    '''Wrapper for KN_eval_request ctypes.Structure
    '''
    _fields_ = [
        ("type", ctypes.c_int),
        ("threadID", ctypes.c_int),
        ("x", ctypes.POINTER (ctypes.c_double)),
        ("lambda_", ctypes.POINTER (ctypes.c_double)),
        ("sigma", ctypes.POINTER (ctypes.c_double)),
        ("vec", ctypes.POINTER (ctypes.c_double)),
    ]
    def __str__(self):
      return "_KN_eval_request"
_KN_eval_request_ptr = ctypes.POINTER (_KN_eval_request)
_KN_eval_request_ptr.__str__ = lambda self: "_KN_eval_request_ptr (" + repr (self.contents) + ")"
_KN_eval_request_ptr.__hash__ = lambda self: str (self.contents).__hash__()

class KN_eval_request:
    def __init__ (self, kc, evalRequest):
        nV            = KN_get_number_vars (kc)
        nC            = KN_get_number_cons (kc)
        self.type     = evalRequest.type
        self.threadID = evalRequest.threadID
        self.x        = _userArray (nV, evalRequest.x)
        self.lambda_  = _userArray (nC+nV, evalRequest.lambda_)
        if evalRequest.sigma:
            self.sigma = evalRequest.sigma.contents.value     # sigma is a simple ctypes.POINTER, not an array
        self.vec = _userArray(nV, evalRequest.vec)
    def __str__ (self):
      return "KN_eval_request"

class _KN_eval_result (ctypes.Structure):
    '''Wrapper for KN_eval_result ctypes.Structure
    '''
    _fields_ = [
        ("obj", ctypes.POINTER (ctypes.c_double)),
        ("c", ctypes.POINTER (ctypes.c_double)),
        ("objGrad", ctypes.POINTER (ctypes.c_double)),
        ("jac", ctypes.POINTER (ctypes.c_double)),
        ("hess", ctypes.POINTER (ctypes.c_double)),
        ("hessVec", ctypes.POINTER (ctypes.c_double)),
        ("rsd", ctypes.POINTER (ctypes.c_double)),
        ("rsdJac", ctypes.POINTER (ctypes.c_double)),
    ]
    def __str__ (self):
      return "_KN_eval_result"
_KN_eval_result_ptr = ctypes.POINTER (_KN_eval_result)
_KN_eval_result_ptr.__str__ = lambda self: "_KN_eval_result_ptr (" + repr (self.contents) + ")"
_KN_eval_result_ptr.__hash__ = lambda self: str (self.contents).__hash__()

class KN_eval_result:
    def __init__ (self, kc, cb, evalResult):
        if evalResult.obj:
            self.obj     = evalResult.obj.contents.value
        self.c       = _userArray (KN_get_cb_number_cons (kc, cb), evalResult.c)
        self.objGrad = _userArray (KN_get_cb_objgrad_nnz (kc, cb), evalResult.objGrad)
        self.jac     = _userArray (KN_get_cb_jacobian_nnz (kc, cb), evalResult.jac)
        self.hess    = _userArray (KN_get_cb_hessian_nnz (kc, cb), evalResult.hess)
        self.hessVec = _userArray (KN_get_number_vars (kc), evalResult.hessVec)
        self.rsd     = _userArray (KN_get_cb_number_rsds (kc, cb), evalResult.rsd)
        self.rsdJac  = _userArray (KN_get_cb_rsd_jacobian_nnz (kc, cb), evalResult.rsdJac)
    def __str__ (self):
      return "KN_eval_result"


#-------------------------------------------------------------------------------
#     PRIVATE FUNCTIONS FOR HANDLING USER CALLBACKS AND PARAMETERS
#-------------------------------------------------------------------------------

#---- Stores a list for each KN_context object, each list storing user parameters (first position) and user callbacks (from second position)
_callbacks = dict ()

def _registerCallback (kc, fnPtr):
    '''Register callback fnPtr into KN_context kc

    This method makes sure that fnPtr is kept alive during the whole
    optimization process and only destroyed upon calling KN_free.
    '''
    if kc not in _callbacks:
        _callbacks[kc] = [None]
    _callbacks[kc].append (fnPtr)

def _registerUserParams (kc, userParams):
    '''Register user parameters userParams into KN_context kc

    This method makes sure that userParams is kept alive during the whole
    optimization process and only destroyed upon calling KN_free.
    '''
    if userParams is not None:
        if kc not in _callbacks:
            _callbacks[kc] = [userParams]
        else:
            _callbacks[kc][0] = userParams

def _unregisterAll (kc):
    '''Unregister all user callbacks and parameters from KN_context kc
    '''
    if kc in _callbacks:
        del _callbacks[kc]

def _getUserParams (kc):
    '''Retrieve the user parameters stored in KN_context kc
    '''
    if kc in _callbacks:
        return _callbacks[kc][0]
    return None


#-------------------------------------------------------------------------------
#     PRIVATE DEFINITIONS AND WRAPPERS FOR USER CALLBACKS
#-------------------------------------------------------------------------------

#---- SIGINT handler
class _signal_handler (object):
    """ This class handles SIGINT to return proper
        KN_RC_USER_TERMINATION to Knitro on interrupt.
    """
    def __init__ (self, fnPtr):
        self._fnPtr = fnPtr

    def call (self, *args, **kwargs):
        try:
            return self._fnPtr (*args, **kwargs)
        except KeyboardInterrupt:
            return KN_RC_USER_TERMINATION
        except ArithmeticError as e:
            print (e)
            return KN_RC_EVAL_ERR
        except Exception as e:
            print (e)
            return KN_RC_CALLBACK_ERR
#---- KN_eval_callback
_KN_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    KN_context_ptr, CB_context_ptr, _KN_eval_request_ptr, _KN_eval_result_ptr, ctypes.c_void_p
)
class _KN_eval_callback_wrapper (object):
    def __init__ (self, kc, fnPtr):
        self._kc = kc
        self._fnPtr = fnPtr
        sig_handler = _signal_handler (self.call)
        self.c_fnPtr = _KN_eval_callback (sig_handler.call)
    def set_KN_context_ptr (self, kc):
        self._kc = kc
    def call (self, kc, cb, c_evalRequest, c_evalResult, c_userParams):
        evalRequest = KN_eval_request (kc, c_evalRequest.contents)
        evalResult = KN_eval_result (kc, cb, c_evalResult.contents)
        ret = self._fnPtr (kc, cb, evalRequest, evalResult, _getUserParams (self._kc))  # TODO which kc to use here?
        if c_evalResult.contents.obj:
            c_evalResult.contents.obj.contents.value = evalResult.obj
        _userToCArray (evalResult.c, c_evalResult.contents.c)
        _userToCArray (evalResult.objGrad, c_evalResult.contents.objGrad)
        _userToCArray (evalResult.jac, c_evalResult.contents.jac)
        _userToCArray (evalResult.hess, c_evalResult.contents.hess)
        _userToCArray (evalResult.hessVec, c_evalResult.contents.hessVec)
        _userToCArray (evalResult.rsd, c_evalResult.contents.rsd)
        _userToCArray (evalResult.rsdJac, c_evalResult.contents.rsdJac)
        return ret

#---- KN_user_callback
_KN_user_callback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    KN_context_ptr,
    ctypes.POINTER (ctypes.c_double), ctypes.POINTER (ctypes.c_double),
    ctypes.c_void_p
)
class _KN_user_callback_wrapper (object):
    def __init__ (self, kc, fnPtr):
        self._kc = kc
        self._fnPtr = fnPtr
        sig_handler = _signal_handler (self.call)
        self.c_fnPtr = _KN_user_callback (sig_handler.call)
    def set_KN_context_ptr (self, kc):
        self._kc = kc
    def call (self, kc, c_x, c_lambda, c_userParams):
        nV = KN_get_number_vars (kc)
        nC = KN_get_number_cons (kc)
        x = _userArray (nV, c_x)
        lambda_ = _userArray (nC+nV, c_lambda)
        ret = self._fnPtr (kc, x, lambda_, _getUserParams (self._kc))
        return ret

#---- KN_ms_initpt_callback
_KN_ms_initpt_callback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    KN_context_ptr,
    ctypes.c_int,
    ctypes.POINTER (ctypes.c_double), ctypes.POINTER (ctypes.c_double),
    ctypes.c_void_p
)
class _KN_ms_initpt_callback_wrapper (object):
    def __init__ (self, kc, fnPtr):
        self._kc = kc
        self._fnPtr = fnPtr
        sig_handler = _signal_handler (self.call)
        self.c_fnPtr = _KN_ms_initpt_callback (sig_handler.call)
    def set_KN_context_ptr (self, kc):
        self._kc = kc
    def call (self, kc, nSolveNumber, c_x, c_lambda, userParams):
        nV = KN_get_number_vars (kc)
        nC = KN_get_number_cons (kc)
        x = _userArray (nV, c_x)
        lambda_ = _userArray (nC+nV, c_lambda)
        ret = self._fnPtr (kc, nSolveNumber, x, lambda_, _getUserParams (self._kc))
        _userToCArray (x, c_x)
        _userToCArray (lambda_, c_lambda)
        return ret

#---- KN_puts
_KN_puts = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)
class _KN_puts_wrapper (object):
    def __init__ (self, kc, fnPtr):
        self._kc = kc
        self._fnPtr = fnPtr
        sig_handler = _signal_handler (self.call)
        self.c_fnPtr = _KN_puts (sig_handler.call)
    def set_KN_context_ptr (self, kc):
        self._kc = kc
    def call (self, str, userParams):
        return self._fnPtr (str.decode ('UTF-8'), _getUserParams (self._kc))


#-------------------------------------------------------------------------------
#     KNITRO RELEASE
#-------------------------------------------------------------------------------

#---- KN_get_release
_knitro.KN_get_release.argtypes = [ctypes.c_int, ctypes.c_char_p]
_knitro.KN_get_release.restype = None
def KN_get_release ():
    length = 15
    c_release = ctypes.c_char_p ((" "*length).encode ('UTF-8'))
    _checkRaise ("KN_get_release", _knitro.KN_get_release (length, c_release))
    return c_release.value.decode ('UTF-8')


#-------------------------------------------------------------------------------
#     CREATING AND DESTROYING KNITRO SOLVER OBJECTS
#-------------------------------------------------------------------------------

def KN_new_init (kc):
    KN_set_int_param(kc, KN_PARAM_PAR_CONCURRENT_EVALS, KN_PAR_CONCURRENT_EVALS_NO)

#---- KN_new
_knitro.KN_new.argtypes = [ctypes.POINTER (KN_context_ptr)]
_knitro.KN_new.restype = ctypes.c_int
def KN_new ():
    kc = KN_context_ptr ()
    _checkRaise ("KN_new", _knitro.KN_new (ctypes.byref (kc)))
    KN_new_init (kc)
    return kc

#---- KN_free
_knitro.KN_free.argtypes = [ctypes.POINTER (KN_context_ptr)]
_knitro.KN_free.restype = ctypes.c_int
def KN_free (kc):
    _unregisterAll (kc)
    _checkRaise ("KN_free", _knitro.KN_free (ctypes.byref (kc)))


#-------------------------------------------------------------------------------
#     CREATING AND DESTROYING KNITRO SOLVER OBJECTS IN HIGH VOLUME
#-------------------------------------------------------------------------------

#---- KN_checkout_license
_knitro.KN_checkout_license.argtypes = [ctypes.POINTER (LM_context_ptr)]
_knitro.KN_checkout_license.restype = ctypes.c_int
def KN_checkout_license ():
    lmc = LM_context_ptr ()
    _checkRaise ("KN_checkout_license", _knitro.KN_checkout_license (ctypes.byref (lmc)))
    return lmc

#---- KN_new_lm
_knitro.KN_new_lm.argtypes = [LM_context_ptr, ctypes.POINTER (KN_context_ptr)]
_knitro.KN_new_lm.restype = KN_context_ptr
def KN_new_lm (lmc):
    kc = KN_context_ptr ()
    _checkRaise ("KN_new_lm", _knitro.KN_new_lm (lmc, ctypes.byref (kc)))
    KN_new_init (kc)
    return kc

#---- KN_release_license
_knitro.KN_release_license.argtypes = [ctypes.POINTER (LM_context_ptr)]
_knitro.KN_release_license.restype = ctypes.c_int
def KN_release_license (lmc):
    _checkRaise ("KN_release_license", _knitro.KN_release_license (ctypes.byref (lmc)))


#-------------------------------------------------------------------------------
#     CHANGING AND READING SOLVER PARAMETERS
#-------------------------------------------------------------------------------

#---- KN_reset_params_to_defaults
_knitro.KN_reset_params_to_defaults.argtypes = [KN_context_ptr]
_knitro.KN_reset_params_to_defaults.restype = ctypes.c_int
def KN_reset_params_to_defaults (kc):
    _checkRaise ("KN_reset_params_to_defaults", _knitro.KN_reset_params_to_defaults (kc))

#---- KN_load_param_file
_knitro.KN_load_param_file.argtypes = [KN_context_ptr, ctypes.c_char_p]
_knitro.KN_load_param_file.restype = ctypes.c_int
def KN_load_param_file (kc, filename):
    _checkRaise ("KN_load_param_file", _knitro.KN_load_param_file (kc, filename.encode ('UTF-8')))

#---- KN_load_tuner_file
_knitro.KN_load_tuner_file.argtypes = [KN_context_ptr, ctypes.c_char_p]
_knitro.KN_load_tuner_file.restype = ctypes.c_int
def KN_load_tuner_file (kc, filename):
    _checkRaise ("KN_load_tuner_file", _knitro.KN_load_tuner_file (kc, filename.encode ('UTF-8')))

#---- KN_save_param_file
_knitro.KN_save_param_file.argtypes = [KN_context_ptr, ctypes.c_char_p]
_knitro.KN_save_param_file.restype = ctypes.c_int
def KN_save_param_file (kc, filename):
    _checkRaise ("KN_save_param_file", _knitro.KN_save_param_file (kc, filename.encode ('UTF-8')))

#---- KN_set_int_param
_knitro.KN_set_int_param_by_name.argtypes = [KN_context_ptr, ctypes.c_char_p, ctypes.c_int]
_knitro.KN_set_int_param_by_name.restype = ctypes.c_int
_knitro.KN_set_int_param.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int]
_knitro.KN_set_int_param.restype = ctypes.c_int
def KN_set_int_param (kc, param, value):
    if isinstance(param, int):
        _checkRaise ("KN_set_int_param", _knitro.KN_set_int_param (kc, param, value))
    else:
        _checkRaise ("KN_set_int_param_by_name", _knitro.KN_set_int_param_by_name (kc, param.encode ('UTF-8'), value))

#---- KN_set_char_param
_knitro.KN_set_char_param_by_name.argtypes = [KN_context_ptr, ctypes.c_char_p, ctypes.c_char_p]
_knitro.KN_set_char_param_by_name.restype = ctypes.c_int
_knitro.KN_set_char_param.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_char_p]
_knitro.KN_set_char_param.restype = ctypes.c_int
def KN_set_char_param (kc, param, value):
    if isinstance(param, int):
        _checkRaise ("KN_set_char_param", _knitro.KN_set_char_param (kc, param, value.encode ('UTF-8')))
    else:
        _checkRaise ("KN_set_char_param_by_name", _knitro.KN_set_char_param_by_name (kc, param.encode ('UTF-8'), value.encode ('UTF-8')))

#---- KN_set_double_param
_knitro.KN_set_double_param_by_name.argtypes = [KN_context_ptr, ctypes.c_char_p, ctypes.c_double]
_knitro.KN_set_double_param_by_name.restype = ctypes.c_int
_knitro.KN_set_double_param.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_double_param.restype = ctypes.c_int
def KN_set_double_param (kc, param, value):
    if isinstance(param, int):
        _checkRaise ("KN_set_double_param", _knitro.KN_set_double_param (kc, param, value))
    else:
        _checkRaise ("KN_set_double_param_by_name", _knitro.KN_set_double_param_by_name (kc, param.encode ('UTF-8'), value))

#---- KN_get_int_param
_knitro.KN_get_int_param_by_name.argtypes = [KN_context_ptr, ctypes.c_char_p, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_int_param_by_name.restype = ctypes.c_int
_knitro.KN_get_int_param.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_int_param.restype = ctypes.c_int
def KN_get_int_param (kc, param):
    c_value = ctypes.c_int (0)
    if isinstance(param, int):
        _checkRaise ("KN_get_int_param", _knitro.KN_get_int_param (kc, param, ctypes.byref (c_value)))
    else:
        _checkRaise ("KN_get_int_param_by_name", _knitro.KN_get_int_param_by_name (kc, param.encode ('UTF-8'), ctypes.byref (c_value)))
    return c_value.value

#---- KN_get_double_param
_knitro.KN_get_double_param_by_name.argtypes = [KN_context_ptr, ctypes.c_char_p, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_double_param_by_name.restype = ctypes.c_int
_knitro.KN_get_double_param.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_double_param.restype = ctypes.c_int
def KN_get_double_param (kc, param):
    c_value = ctypes.c_double (0)
    if isinstance(param, int):
        _checkRaise ("KN_get_double_param", _knitro.KN_get_double_param (kc, param, ctypes.byref (c_value)))
    else:
        _checkRaise ("KN_get_double_param_by_name", _knitro.KN_get_double_param_by_name (kc, param.encode ('UTF-8'), ctypes.byref (c_value)))
    return c_value.value

#---- KN_get_param_name
_knitro.KN_get_param_name.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
_knitro.KN_get_param_name.restype = ctypes.c_int
def KN_get_param_name (kc, param_id):
    output_size = 128
    c_param_name = ctypes.c_char_p ((" "*output_size).encode ('UTF-8'))
    _checkRaise ("KN_get_param_name", _knitro.KN_get_param_name (kc, param_id, c_param_name, output_size))
    return c_param_name.value.decode ('UTF-8')

#---- KN_get_param_doc
_knitro.KN_get_param_doc.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
_knitro.KN_get_param_doc.restype = ctypes.c_int
def KN_get_param_doc (kc, param_id):
    output_size = 128
    c_description = ctypes.c_char_p ((" "*output_size).encode ('UTF-8'))
    _checkRaise ("KN_get_param_doc", _knitro.KN_get_param_doc (kc, param_id, c_description, output_size))
    return c_description.value.decode ('UTF-8')

#---- KN_get_param_type
_knitro.KN_get_param_type.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_param_type.restype = ctypes.c_int
def KN_get_param_type (kc, param_id):
    c_param_type = ctypes.c_int (0)
    _checkRaise ("KN_get_param_type", _knitro.KN_get_param_type (kc, param_id, ctypes.byref (c_param_type)))
    return c_param_type.value

#---- KN_get_num_param_values
_knitro.KN_get_num_param_values.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_num_param_values.restype = ctypes.c_int
def KN_get_num_param_values (kc, param_id):
    c_num_param_values = ctypes.c_int (0)
    _checkRaise ("KN_get_num_param_values", _knitro.KN_get_num_param_values (kc, param_id, ctypes.byref (c_num_param_values)))
    return c_num_param_values.value

#---- KN_get_param_value_doc
_knitro.KN_get_param_value_doc.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
_knitro.KN_get_param_value_doc.restype = ctypes.c_int
def KN_get_param_value_doc (kc, param_id, value_id):
    output_size = 128
    c_param_value_string = ctypes.c_char_p ((" "*output_size).encode ('UTF-8'))
    _checkRaise ("KN_get_param_value_doc", _knitro.KN_get_param_value_doc (kc, param_id, value_id, c_param_value_string, output_size))
    return c_param_value_string.value.decode ('UTF-8')

#---- KN_get_param_id
_knitro.KN_get_param_id.argtypes = [KN_context_ptr, ctypes.c_char_p, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_param_id.restype = ctypes.c_int
def KN_get_param_id (kc, name):
    c_param_id = ctypes.c_int (0)
    _checkRaise ("KN_get_param_id", _knitro.KN_get_param_id (kc, name.encode ('UTF-8'), ctypes.byref (c_param_id)))
    return c_param_id.value


#-------------------------------------------------------------------------------
#     PROBLEM CONSTRUCTION
#-------------------------------------------------------------------------------

#---- KN_add_vars
_knitro.KN_add_vars.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_add_vars.restype = ctypes.c_int
def KN_add_vars (kc, nV):
    c_indexVars = (ctypes.c_int * nV) ()
    _checkRaise ("KN_add_vars", _knitro.KN_add_vars (kc, nV, c_indexVars))
    return _userArray(nV, c_indexVars)

#---- KN_add_var
_knitro.KN_add_var.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_add_var.restype = ctypes.c_int
def KN_add_var (kc):
    c_indexVar = ctypes.c_int (0)
    _checkRaise ("KN_add_var", _knitro.KN_add_var (kc, ctypes.byref (c_indexVar)))
    return c_indexVar.value

#---- KN_add_cons
_knitro.KN_add_cons.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_add_cons.restype = ctypes.c_int
def KN_add_cons (kc, nC):
    c_indexCons = (ctypes.c_int * nC) ()
    _checkRaise ("KN_add_cons", _knitro.KN_add_cons (kc, nC, c_indexCons))
    return _userArray(nC, c_indexCons)

#---- KN_add_con
_knitro.KN_add_con.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_add_con.restype = ctypes.c_int
def KN_add_con (kc):
    c_indexCon = ctypes.c_int (0)
    _checkRaise ("KN_add_con", _knitro.KN_add_con (kc, ctypes.byref (c_indexCon)))
    return c_indexCon.value

#---- KN_add_rsds
_knitro.KN_add_rsds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_add_rsds.restype = ctypes.c_int
def KN_add_rsds (kc, nR):
    c_indexRsds = (ctypes.c_int * nR) ()
    _checkRaise ("KN_add_rsds", _knitro.KN_add_rsds (kc, nR, c_indexRsds))
    return _userArray(nR, c_indexRsds)

#---- KN_add_rsd
_knitro.KN_add_rsd.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_add_rsd.restype = ctypes.c_int
def KN_add_rsd (kc):
    c_indexRsd = ctypes.c_int (0)
    _checkRaise ("KN_add_rsd", _knitro.KN_add_rsd (kc, ctypes.byref (c_indexRsd)))
    return c_indexRsd.value

#---- KN_set_var_lobnds
_knitro.KN_set_var_lobnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_lobnds.restype = ctypes.c_int
_knitro.KN_set_var_lobnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_lobnds_all.restype = ctypes.c_int
_knitro.KN_set_var_lobnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_var_lobnd.restype = ctypes.c_int
def KN_set_var_lobnds (kc, indexVars = None, xLoBnds = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_lobnds_all", _knitro.KN_set_var_lobnds_all (kc, _cDoubleArray (xLoBnds)))
    else:
        try:
            nV = len (indexVars)
            if xLoBnds is None or nV != len (xLoBnds):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xLoBnds have different sizes!")
            _checkRaise ("KN_set_var_lobnds", _knitro.KN_set_var_lobnds (kc, nV, _cIntArray (indexVars), _cDoubleArray (xLoBnds)))
        except TypeError:
            _checkRaise ("KN_set_var_lobnd", _knitro.KN_set_var_lobnd (kc, indexVars, xLoBnds))

#---- KN_get_var_lobnds
_knitro.KN_get_var_lobnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_var_lobnds.restype = ctypes.c_int
_knitro.KN_get_var_lobnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_var_lobnds_all.restype = ctypes.c_int
_knitro.KN_get_var_lobnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_knitro.KN_get_var_lobnd.restype = ctypes.c_int
def KN_get_var_lobnds(kc, indexVars=None):
    if indexVars is None:
        nC = KN_get_number_vars(kc)
        lobnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_var_lobnds_all", _knitro.KN_get_var_lobnds_all(kc, lobnds))
        return _userArray(nC, lobnds)
    elif isiterable(indexVars):
        nC = len(indexVars)
        lobnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_var_lobnds", _knitro.KN_get_var_lobnds(kc, nC, _cIntArray(indexVars), lobnds))
        return _userArray(nC, lobnds)
    elif isinstance(indexVars, int):
        lobnd = ctypes.c_double (0)
        _checkRaise ("KN_get_var_lobnd", _knitro.KN_get_var_lobnd(kc, indexVars, ctypes.byref(lobnd)))
        return lobnd.value
    else:
        raise TypeError("Error in KN_get_var_lobnds: indexVars sould be"
                        " None, int or list")

#---- KN_set_var_upbnds
_knitro.KN_set_var_upbnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_upbnds.restype = ctypes.c_int
_knitro.KN_set_var_upbnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_upbnds_all.restype = ctypes.c_int
_knitro.KN_set_var_upbnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_var_upbnd.restype = ctypes.c_int
def KN_set_var_upbnds (kc, indexVars = None, xUpBnds = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_upbnds_all", _knitro.KN_set_var_upbnds_all (kc, _cDoubleArray (xUpBnds)))
    else:
        try:
            nV = len (indexVars)
            if xUpBnds is None or nV != len (xUpBnds):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xUpBnds have different sizes!")
            _checkRaise ("KN_set_var_upbnds", _knitro.KN_set_var_upbnds (kc, nV, _cIntArray (indexVars), _cDoubleArray (xUpBnds)))
        except TypeError:
            _checkRaise ("KN_set_var_upbnd", _knitro.KN_set_var_upbnd (kc, indexVars, xUpBnds))

#---- KN_get_var_upbnds
_knitro.KN_get_var_upbnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_var_upbnds.restype = ctypes.c_int
_knitro.KN_get_var_upbnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_var_upbnds_all.restype = ctypes.c_int
_knitro.KN_get_var_upbnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_knitro.KN_get_var_upbnd.restype = ctypes.c_int
def KN_get_var_upbnds(kc, indexVars=None):
    if indexVars is None:
        nC = KN_get_number_vars(kc)
        upbnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_var_upbnds_all", _knitro.KN_get_var_upbnds_all(kc, upbnds))
        return _userArray(nC, upbnds)
    elif isiterable(indexVars):
        nC = len(indexVars)
        upbnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_var_upbnds", _knitro.KN_get_var_upbnds(kc, nC, _cIntArray(indexVars), upbnds))
        return _userArray(nC, upbnds)
    elif isinstance(indexVars, int):
        upbnd = ctypes.c_double (0)
        _checkRaise ("KN_get_var_upbnd", _knitro.KN_get_var_upbnd(kc, indexVars, ctypes.byref(upbnd)))
        return upbnd.value
    else:
        raise TypeError("Error in KN_get_var_upbnds: indexVars sould be"
                        " None, int or list")

#---- KN_set_var_fxbnds
_knitro.KN_set_var_fxbnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_fxbnds.restype = ctypes.c_int
_knitro.KN_set_var_fxbnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_fxbnds_all.restype = ctypes.c_int
_knitro.KN_set_var_fxbnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_var_fxbnd.restype = ctypes.c_int
def KN_set_var_fxbnds (kc, indexVars = None, xFxBnds = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_fxbnds_all", _knitro.KN_set_var_fxbnds_all (kc, _cDoubleArray (xFxBnds)))
    else:
        try:
            nV = len (indexVars)
            if xFxBnds is None or nV != len (xFxBnds):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xFxBnds have different sizes!")
            _checkRaise ("KN_set_var_fxbnds", _knitro.KN_set_var_fxbnds (kc, nV, _cIntArray (indexVars), _cDoubleArray (xFxBnds)))
        except TypeError:
            _checkRaise ("KN_set_var_fxbnd", _knitro.KN_set_var_fxbnd (kc, indexVars, xFxBnds))

#---- KN_get_var_fxbnds
_knitro.KN_get_var_fxbnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_var_fxbnds.restype = ctypes.c_int
_knitro.KN_get_var_fxbnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_var_fxbnds_all.restype = ctypes.c_int
_knitro.KN_get_var_fxbnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_knitro.KN_get_var_fxbnd.restype = ctypes.c_int
def KN_get_var_fxbnds(kc, indexVars=None):
    if indexVars is None:
        nC = KN_get_number_vars(kc)
        fxbnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_var_fxbnds_all", _knitro.KN_get_var_fxbnds_all(kc, fxbnds))
        return _userArray(nC, fxbnds)
    elif isiterable(indexVars):
        nC = len(indexVars)
        fxbnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_var_fxbnds", _knitro.KN_get_var_fxbnds(kc, nC, _cIntArray(indexVars), fxbnds))
        return _userArray(nC, fxbnds)
    elif isinstance(indexVars, int):
        fxbnd = ctypes.c_double (0)
        _checkRaise ("KN_get_var_fxbnd", _knitro.KN_get_var_fxbnd(kc, indexVars, ctypes.byref(fxbnd)))
        return fxbnd.value
    else:
        raise TypeError("Error in KN_get_var_fxbnds: indexVars sould be"
                        " None, int or list")

#---- KN_set_var_types
_knitro.KN_set_var_types.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_var_types.restype = ctypes.c_int
_knitro.KN_set_var_types_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_var_types_all.restype = ctypes.c_int
_knitro.KN_set_var_type.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int]
_knitro.KN_set_var_type.restype = ctypes.c_int
def KN_set_var_types (kc, indexVars = None, xTypes = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_types_all", _knitro.KN_set_var_types_all (kc, _cIntArray (xTypes)))
    else:
        try:
            nV = len (indexVars)
            if xTypes is None or nV != len (xTypes):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xTypes have different sizes!")
            _checkRaise ("KN_set_var_types", _knitro.KN_set_var_types (kc, nV, _cIntArray (indexVars), _cIntArray (xTypes)))
        except TypeError:
            _checkRaise ("KN_set_var_type", _knitro.KN_set_var_type (kc, indexVars, xTypes))

#---- KN_set_var_properties
_knitro.KN_set_var_properties.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_var_properties.restype = ctypes.c_int
_knitro.KN_set_var_properties_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_var_properties_all.restype = ctypes.c_int
_knitro.KN_set_var_property.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int]
_knitro.KN_set_var_property.restype = ctypes.c_int
def KN_set_var_properties (kc, indexVars = None, xProperties = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_properties_all", _knitro.KN_set_var_properties_all (kc, _cIntArray (xProperties)))
    else:
        try:
            nV = len (indexVars)
            if xProperties is None or nV != len (xProperties):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xProperties have different sizes!")
            _checkRaise ("KN_set_var_properties", _knitro.KN_set_var_properties (kc, nV, _cIntArray (indexVars), _cIntArray (xProperties)))
        except TypeError:
            _checkRaise ("KN_set_var_property", _knitro.KN_set_var_property (kc, indexVars, xProperties))

#---- KN_set_con_lobnds
_knitro.KN_set_con_lobnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_lobnds.restype = ctypes.c_int
_knitro.KN_set_con_lobnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_lobnds_all.restype = ctypes.c_int
_knitro.KN_set_con_lobnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_con_lobnd.restype = ctypes.c_int
def KN_set_con_lobnds (kc, indexCons = None, cLoBnds = None):
    if indexCons is None:
        _checkRaise ("KN_set_con_lobnds_all", _knitro.KN_set_con_lobnds_all (kc, _cDoubleArray (cLoBnds)))
    else:
        try:
            nC = len (indexCons)
            if cLoBnds is None or nC != len (cLoBnds):
                raise ValueError ("Knitro-Python error: Arrays indexCons and cLoBnds have different sizes!")
            _checkRaise ("KN_set_con_lobnds", _knitro.KN_set_con_lobnds (kc, nC, _cIntArray (indexCons), _cDoubleArray (cLoBnds)))
        except TypeError:
            _checkRaise ("KN_set_con_lobnd", _knitro.KN_set_con_lobnd (kc, indexCons, cLoBnds))

#---- KN_get_con_lobnds
_knitro.KN_get_con_lobnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_con_lobnds.restype = ctypes.c_int
_knitro.KN_get_con_lobnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_con_lobnds_all.restype = ctypes.c_int
_knitro.KN_get_con_lobnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_knitro.KN_get_con_lobnd.restype = ctypes.c_int
def KN_get_con_lobnds(kc, indexCons=None):
    if indexCons is None:
        nC = KN_get_number_cons(kc)
        lobnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_con_lobnds_all", _knitro.KN_get_con_lobnds_all(kc, lobnds))
        return _userArray(nC, lobnds)
    elif isiterable(indexCons):
        nC = len(indexCons)
        lobnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_con_lobnds", _knitro.KN_get_con_lobnds(kc, nC, _cIntArray(indexCons), lobnds))
        return _userArray(nC, lobnds)
    elif isinstance(indexCons, int):
        lobnd = ctypes.c_double (0)
        _checkRaise ("KN_get_con_lobnd", _knitro.KN_get_con_lobnd(kc, indexCons, ctypes.byref(lobnd)))
        return lobnd.value
    else:
        raise TypeError("Error in KN_get_con_lobnds: indexCons sould be"
                        " None, int or list")

#---- KN_set_con_upbnds
_knitro.KN_set_con_upbnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_upbnds.restype = ctypes.c_int
_knitro.KN_set_con_upbnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_upbnds_all.restype = ctypes.c_int
_knitro.KN_set_con_upbnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_con_upbnd.restype = ctypes.c_int
def KN_set_con_upbnds (kc, indexCons = None, cUpBnds = None):
    if indexCons is None:
        _checkRaise ("KN_set_con_upbnds_all", _knitro.KN_set_con_upbnds_all (kc, _cDoubleArray (cUpBnds)))
    else:
        try:
            nC = len (indexCons)
            if cUpBnds is None or nC != len (cUpBnds):
                raise ValueError ("Knitro-Python error: Arrays indexCons and cUpBnds have different sizes!")
            _checkRaise ("KN_set_con_upbnds", _knitro.KN_set_con_upbnds (kc, nC, _cIntArray (indexCons), _cDoubleArray (cUpBnds)))
        except TypeError:
            _checkRaise ("KN_set_con_upbnd", _knitro.KN_set_con_upbnd (kc, indexCons, cUpBnds))

#---- KN_get_con_upbnds
_knitro.KN_get_con_upbnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_con_upbnds.restype = ctypes.c_int
_knitro.KN_get_con_upbnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_con_upbnds_all.restype = ctypes.c_int
_knitro.KN_get_con_upbnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_knitro.KN_get_con_upbnd.restype = ctypes.c_int
def KN_get_con_upbnds(kc, indexCons=None):
    if indexCons is None:
        nC = KN_get_number_cons(kc)
        upbnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_con_upbnds_all", _knitro.KN_get_con_upbnds_all(kc, upbnds))
        return _userArray(nC, upbnds)
    elif isiterable(indexCons):
        nC = len(indexCons)
        upbnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_con_upbnds", _knitro.KN_get_con_upbnds(kc, nC, _cIntArray(indexCons), upbnds))
        return _userArray(nC, upbnds)
    elif isinstance(indexCons, int):
        upbnd = ctypes.c_double (0)
        _checkRaise ("KN_get_con_upbnd", _knitro.KN_get_con_upbnd(kc, indexCons, ctypes.byref(upbnd)))
        return upbnd.value
    else:
        raise TypeError("Error in KN_get_con_upbnds: indexCons sould be"
                        " None, int or list")


#---- KN_set_con_eqbnds
_knitro.KN_set_con_eqbnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_eqbnds.restype = ctypes.c_int
_knitro.KN_set_con_eqbnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_eqbnds_all.restype = ctypes.c_int
_knitro.KN_set_con_eqbnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_con_eqbnd.restype = ctypes.c_int
def KN_set_con_eqbnds (kc, indexCons = None, cEqBnds = None):
    if indexCons is None:
        _checkRaise ("KN_set_con_eqbnds_all", _knitro.KN_set_con_eqbnds_all (kc, _cDoubleArray (cEqBnds)))
    else:
        try:
            nC = len (indexCons)
            if cEqBnds is None or nC != len (cEqBnds):
                raise ValueError ("Knitro-Python error: Arrays indexCons and cEqBnds have different sizes!")
            _checkRaise ("KN_set_con_eqbnds", _knitro.KN_set_con_eqbnds (kc, nC, _cIntArray (indexCons), _cDoubleArray (cEqBnds)))
        except TypeError:
            _checkRaise ("KN_set_con_eqbnd", _knitro.KN_set_con_eqbnd (kc, indexCons, cEqBnds))

#---- KN_get_con_eqbnds
_knitro.KN_get_con_eqbnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_con_eqbnds.restype = ctypes.c_int
_knitro.KN_get_con_eqbnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_con_eqbnds_all.restype = ctypes.c_int
_knitro.KN_get_con_eqbnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_knitro.KN_get_con_eqbnd.restype = ctypes.c_int
def KN_get_con_eqbnds(kc, indexCons=None):
    if indexCons is None:
        nC = KN_get_number_cons(kc)
        eqbnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_con_eqbnds_all", _knitro.KN_get_con_eqbnds_all(kc, eqbnds))
        return _userArray(nC, eqbnds)
    elif isiterable(indexCons):
        nC = len(indexCons)
        eqbnds = (ctypes.c_double * nC)()
        _checkRaise ("KN_get_con_eqbnds", _knitro.KN_get_con_eqbnds(kc, nC, _cIntArray(indexCons), eqbnds))
        return _userArray(nC, eqbnds)
    elif isinstance(indexCons, int):
        eqbnd = ctypes.c_double (0)
        _checkRaise ("KN_get_con_eqbnd", _knitro.KN_get_con_eqbnd(kc, indexCons, ctypes.byref(eqbnd)))
        return eqbnd.value
    else:
        raise TypeError("Error in KN_get_con_eqbnds: indexCons sould be"
                        " None, int or list")


#---- KN_set_obj_property
_knitro.KN_set_obj_property.argtypes = [KN_context_ptr, ctypes.c_int]
_knitro.KN_set_obj_property.restype = ctypes.c_int
def KN_set_obj_property (kc, objProperty):
    _checkRaise ("KN_set_obj_property", _knitro.KN_set_obj_property (kc, objProperty))

#---- KN_set_con_properties
_knitro.KN_set_con_properties.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_con_properties.restype = ctypes.c_int
_knitro.KN_set_con_properties_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_con_properties_all.restype = ctypes.c_int
_knitro.KN_set_con_property.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int]
_knitro.KN_set_con_property.restype = ctypes.c_int
def KN_set_con_properties (kc, indexCons = None, cProperties = None):
    if indexCons is None:
        _checkRaise ("KN_set_con_properties_all", _knitro.KN_set_con_properties_all (kc, _cIntArray (cProperties)))
    else:
        try:
            nC = len (indexCons)
            if cProperties is None or nC != len (cProperties):
                raise ValueError ("Knitro-Python error: Arrays indexCons and cProperties have different sizes!")
            _checkRaise ("KN_set_con_properties", _knitro.KN_set_con_properties (kc, nC, _cIntArray (indexCons), _cIntArray (cProperties)))
        except TypeError:
            _checkRaise ("KN_set_con_property", _knitro.KN_set_con_property (kc, indexCons, cProperties))

#---- KN_set_obj_goal
_knitro.KN_set_obj_goal.argtypes = [KN_context_ptr, ctypes.c_int]
_knitro.KN_set_obj_goal.restype = ctypes.c_int
def KN_set_obj_goal (kc, objGoal):
    _checkRaise ("KN_set_obj_goal", _knitro.KN_set_obj_goal (kc, objGoal))

#---- KN_set_var_primal_init_values
_knitro.KN_set_var_primal_init_values.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_primal_init_values.restype = ctypes.c_int
_knitro.KN_set_var_primal_init_values_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_primal_init_values_all.restype = ctypes.c_int
_knitro.KN_set_var_primal_init_value.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_var_primal_init_value.restype = ctypes.c_int
def KN_set_var_primal_init_values (kc, indexVars = None, xInitVals = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_primal_init_values_all", _knitro.KN_set_var_primal_init_values_all (kc, _cDoubleArray (xInitVals)))
    else:
        try:
            nV = len (indexVars)
            if xInitVals is None or nV != len (xInitVals):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xInitVals have different sizes!")
            _checkRaise ("KN_set_var_primal_init_values", _knitro.KN_set_var_primal_init_values (kc, nV, _cIntArray (indexVars), _cDoubleArray (xInitVals)))
        except TypeError:
            _checkRaise ("KN_set_var_primal_init_value", _knitro.KN_set_var_primal_init_value (kc, indexVars, xInitVals))

#---- KN_set_var_dual_init_values
_knitro.KN_set_var_dual_init_values.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_dual_init_values.restype = ctypes.c_int
_knitro.KN_set_var_dual_init_values_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_dual_init_values_all.restype = ctypes.c_int
_knitro.KN_set_var_dual_init_value.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_var_dual_init_value.restype = ctypes.c_int
def KN_set_var_dual_init_values (kc, indexVars = None, lambdaInitVals = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_dual_init_values_all", _knitro.KN_set_var_dual_init_values_all (kc, _cDoubleArray (lambdaInitVals)))
    else:
        try:
            nV = len (indexVars)
            if lambdaInitVals is None or nV != len (lambdaInitVals):
                raise ValueError ("Knitro-Python error: Arrays indexVars and lambdaInitVals have different sizes!")
            _checkRaise ("KN_set_var_dual_init_values", _knitro.KN_set_var_dual_init_values (kc, nV, _cIntArray (indexVars), _cDoubleArray (lambdaInitVals)))
        except TypeError:
            _checkRaise ("KN_set_var_dual_init_value", _knitro.KN_set_var_dual_init_value (kc, indexVars, lambdaInitVals))

#---- KN_set_con_dual_init_values
_knitro.KN_set_con_dual_init_values.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_dual_init_values.restype = ctypes.c_int
_knitro.KN_set_con_dual_init_values_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_dual_init_values_all.restype = ctypes.c_int
_knitro.KN_set_con_dual_init_value.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_con_dual_init_value.restype = ctypes.c_int
def KN_set_con_dual_init_values (kc, indexCons = None, lambdaInitVals = None):
    if indexCons is None:
        _checkRaise ("KN_set_con_dual_init_values_all", _knitro.KN_set_con_dual_init_values_all (kc, _cDoubleArray (lambdaInitVals)))
    else:
        try:
            nC = len (indexCons)
            if lambdaInitVals is None or nC != len (lambdaInitVals):
                raise ValueError ("Knitro-Python error: Arrays indexCons and lambdaInitVals have different sizes!")
            _checkRaise ("KN_set_con_dual_init_values", _knitro.KN_set_con_dual_init_values (kc, nC, _cIntArray (indexCons), _cDoubleArray (lambdaInitVals)))
        except TypeError:
            _checkRaise ("KN_set_con_dual_init_value", _knitro.KN_set_con_dual_init_value (kc, indexCons, lambdaInitVals))

#---- KN_load_mps_file
_knitro.KN_load_mps_file.argtypes = [KN_context_ptr, ctypes.c_char_p]
_knitro.KN_load_mps_file.restype = ctypes.c_int
def KN_load_mps_file(kc, filename):
    _checkRaise ("KN_load_mps_file", _knitro.KN_load_mps_file (kc, filename.encode ('UTF-8')))

#---- KN_write_mps_file
_knitro.KN_write_mps_file.argtypes = [KN_context_ptr, ctypes.c_char_p]
_knitro.KN_write_mps_file.restype = ctypes.c_int
def KN_write_mps_file(kc, filename):
    _checkRaise ("KN_write_mps_file", _knitro.KN_write_mps_file (kc, filename.encode ('UTF-8')))

#-------------------------------------------------------------------------------
#     ADDING CONSTANT STRUCTURE
#-------------------------------------------------------------------------------

#---- KN_add_obj_constant
_knitro.KN_add_obj_constant.argtypes = [KN_context_ptr, ctypes.c_double]
_knitro.KN_add_obj_constant.restype = ctypes.c_int
def KN_add_obj_constant (kc, constant):
    _checkRaise ("KN_add_obj_constant", _knitro.KN_add_obj_constant (kc, constant))

#---- KN_add_con_constants
_knitro.KN_add_con_constants.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_con_constants.restype = ctypes.c_int
_knitro.KN_add_con_constants_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_con_constants_all.restype = ctypes.c_int
_knitro.KN_add_con_constant.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_add_con_constant.restype = ctypes.c_int
def KN_add_con_constants (kc, indexCons = None, constants = None):
    if indexCons is None:
        _checkRaise ("KN_add_con_constants_all", _knitro.KN_add_con_constants_all (kc, _cDoubleArray (constants)))
    else:
        try:
            nC = len (indexCons)
            if constants is None or nC != len (constants):
                raise ValueError ("Knitro-Python error: Arrays indexCons and constants have different sizes!")
            _checkRaise ("KN_add_con_constants", _knitro.KN_add_con_constants (kc, nC, _cIntArray (indexCons), _cDoubleArray (constants)))
        except TypeError:
            _checkRaise ("KN_add_con_constant", _knitro.KN_add_con_constant (kc, indexCons, constants))

#---- KN_add_rsd_constants
_knitro.KN_add_rsd_constants.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_rsd_constants.restype = ctypes.c_int
_knitro.KN_add_rsd_constants_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_rsd_constants_all.restype = ctypes.c_int
_knitro.KN_add_rsd_constant.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_add_rsd_constant.restype = ctypes.c_int
def KN_add_rsd_constants (kc, indexRsds = None, constants = None):
    if indexRsds is None:
        _checkRaise ("KN_add_rsd_constants_all", _knitro.KN_add_rsd_constants_all (kc, _cDoubleArray (constants)))
    else:
        try:
            nC = len (indexRsds)
            if constants is None or nC != len (constants):
                raise ValueError ("Knitro-Python error: Arrays indexRsds and constants have different sizes!")
            _checkRaise ("KN_add_rsd_constants", _knitro.KN_add_rsd_constants (kc, nC, _cIntArray (indexRsds), _cDoubleArray (constants)))
        except TypeError:
            _checkRaise ("KN_add_rsd_constant", _knitro.KN_add_rsd_constant (kc, indexCons, constants))


#-------------------------------------------------------------------------------
#     ADDING LINEAR STRUCTURE
#-------------------------------------------------------------------------------

#---- KN_add_obj_linear_struct
_knitro.KN_add_obj_linear_struct.argtypes = [KN_context_ptr, KNLONG, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_obj_linear_struct.restype = ctypes.c_int
def KN_add_obj_linear_struct (kc, indexVars, coefs):
    if indexVars is not None:
        try:
            nnz = len (indexVars)
            if coefs is None or nnz != len (coefs):
                raise ValueError ("Knitro-Python error: Arrays indexVars and coefs have different sizes!")
            _checkRaise ("KN_add_obj_linear_struct", _knitro.KN_add_obj_linear_struct (kc, nnz, _cIntArray (indexVars), _cDoubleArray (coefs)))
        except TypeError:
            _checkRaise ("KN_add_obj_linear_struct", _knitro.KN_add_obj_linear_struct (kc, 1, _cIntPointer (indexVars), _cDoublePointer (coefs)))

#---- KN_add_con_linear_struct
_knitro.KN_add_con_linear_struct.argtypes = [KN_context_ptr, KNLONG, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_con_linear_struct.restype = ctypes.c_int
_knitro.KN_add_con_linear_struct_one.argtypes = [KN_context_ptr, KNLONG, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_con_linear_struct_one.restype = ctypes.c_int
def KN_add_con_linear_struct (kc, indexCons, indexVars, coefs):
    try:
        nnz = len (indexCons)
        if indexVars is None or nnz != len (indexVars) or coefs is None or nnz != len (coefs):
            raise ValueError ("Knitro-Python error: Arrays indexCons, indexVars and coefs have different sizes!")
        _checkRaise ("KN_add_con_linear_struct", _knitro.KN_add_con_linear_struct (kc, nnz, _cIntArray (indexCons), _cIntArray (indexVars), _cDoubleArray (coefs)))
    except TypeError:
        if indexVars is not None:
            try:
                nnz = len (indexVars)
                if coefs is None or nnz != len (coefs):
                    raise ValueError ("Knitro-Python error: Arrays indexVars and coefs have different sizes!")
                _checkRaise ("KN_add_con_linear_struct_one", _knitro.KN_add_con_linear_struct_one (kc, nnz, indexCons, _cIntArray (indexVars), _cDoubleArray (coefs)))
            except TypeError:
                _checkRaise ("KN_add_con_linear_struct_one", _knitro.KN_add_con_linear_struct_one (kc, 1, indexCons, _cIntPointer (indexVars), _cDoublePointer (coefs)))

#---- KN_add_rsd_linear_struct
_knitro.KN_add_rsd_linear_struct.argtypes = [KN_context_ptr, KNLONG, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_rsd_linear_struct.restype = ctypes.c_int
_knitro.KN_add_rsd_linear_struct_one.argtypes = [KN_context_ptr, KNLONG, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_rsd_linear_struct_one.restype = ctypes.c_int
def KN_add_rsd_linear_struct (kc, indexRsds, indexVars, coefs):
    try:
        nnz = len (indexRsds)
        if indexVars is None or nnz != len (indexVars) or coefs is None or nnz != len (coefs):
            raise ValueError ("Knitro-Python error: Arrays indexRsds, indexVars and coefs have different sizes!")
        _checkRaise ("KN_add_rsd_linear_struct", _knitro.KN_add_rsd_linear_struct (kc, nnz, _cIntArray (indexRsds), _cIntArray (indexVars), _cDoubleArray (coefs)))
    except TypeError:
        try:
            if indexVars is not None:
                nnz = len (indexVars)
                if coefs is None or nnz != len (coefs):
                    raise ValueError ("Knitro-Python error: Arrays indexVars and coefs have different sizes!")
                _checkRaise ("KN_add_rsd_linear_struct_one", _knitro.KN_add_rsd_linear_struct_one (kc, nnz, indexRsds, _cIntArray (indexVars), _cDoubleArray (coefs)))
        except TypeError:
            _checkRaise ("KN_add_rsd_linear_struct_one", _knitro.KN_add_rsd_linear_struct_one (kc, 1, indexRsds, _cIntPointer (indexVars), _cDoublePointer (coefs)))

#-------------------------------------------------------------------------------
#     ADDING QUADRATIC STRUCTURE
#-------------------------------------------------------------------------------

#---- KN_add_obj_quadratic_struct
_knitro.KN_add_obj_quadratic_struct.argtypes = [KN_context_ptr, KNLONG, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_obj_quadratic_struct.restype = ctypes.c_int
def KN_add_obj_quadratic_struct (kc, indexVars1, indexVars2, coefs):
    try:
        nnz = len (indexVars1)
        if indexVars2 is None or nnz != len (indexVars2) or coefs is None or nnz != len (coefs):
            raise ValueError ("Knitro-Python error: Arrays indexVars1, indexVars2 and coefs have different sizes!")
        _checkRaise (
            "KN_add_obj_quadratic_struct",
            _knitro.KN_add_obj_quadratic_struct (kc, nnz, _cIntArray (indexVars1), _cIntArray (indexVars2), _cDoubleArray (coefs))
        )
    except TypeError:
        _checkRaise (
            "KN_add_obj_quadratic_struct",
            _knitro.KN_add_obj_quadratic_struct (kc, 1, _cIntPointer (indexVars1), _cIntPointer (indexVars2), _cDoublePointer (coefs))
        )


#---- KN_add_con_quadratic_struct
_knitro.KN_add_con_quadratic_struct.argtypes = [KN_context_ptr, KNLONG, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_con_quadratic_struct.restype = ctypes.c_int
_knitro.KN_add_con_quadratic_struct_one.argtypes = [KN_context_ptr, KNLONG, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_add_con_quadratic_struct_one.restype = ctypes.c_int
def KN_add_con_quadratic_struct (kc, indexCons, indexVars1, indexVars2, coefs):
    try:
        nnz = len (indexCons)
        if indexVars1 is None or nnz != len (indexVars1) or indexVars2 is None or nnz != len (indexVars2) or coefs is None or nnz != len (coefs):
            raise ValueError ("Knitro-Python error: Arrays indexCons, indexVars1, indexVars2 and coefs have different sizes!")
        _checkRaise (
            "KN_add_con_quadratic_struct",
            _knitro.KN_add_con_quadratic_struct (kc, nnz, _cIntArray (indexCons), _cIntArray (indexVars1), _cIntArray (indexVars2), _cDoubleArray (coefs))
        )
    except TypeError:
        try:
            nnz = len (indexVars1)
            if indexVars2 is None or nnz != len (indexVars2) or coefs is None or nnz != len (coefs):
                raise ValueError ("Knitro-Python error: Arrays indexVars1, indexVars2 and coefs have different sizes!")
            _checkRaise (
                "KN_add_con_quadratic_struct_one",
                _knitro.KN_add_con_quadratic_struct_one (kc, nnz, indexCons, _cIntArray (indexVars1), _cIntArray (indexVars2), _cDoubleArray (coefs))
            )
        except TypeError:
            _checkRaise (
                "KN_add_con_quadratic_struct_one",
                _knitro.KN_add_con_quadratic_struct_one (kc, 1, indexCons, _cIntPointer (indexVars1), _cIntPointer (indexVars2), _cDoublePointer (coefs))
            )


#-------------------------------------------------------------------------------
#     ADDING CONIC STRUCTURE
#-------------------------------------------------------------------------------

#---- KN_add_con_L2norm
_knitro.KN_add_con_L2norm.argtypes = [
    KN_context_ptr, ctypes.c_int, ctypes.c_int,
    KNLONG, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double),
    ctypes.POINTER (ctypes.c_double)
]
_knitro.KN_add_con_L2norm.restype = ctypes.c_int
def KN_add_con_L2norm (kc, indexCon, nCoords, indexCoords, indexVars, coefs, constants = None):
    if indexVars is not None:
        nnz = len (indexVars)
        if (indexCoords is None or nnz != len (indexCoords)) and (coefs is None or nnz != len (coefs)):
            raise ValueError ("Knitro-Python error: Arrays indexCoords, indexVars and coefs have different sizes!")
        _checkRaise (
            "KN_add_con_L2norm",
            _knitro.KN_add_con_L2norm (kc, indexCon, nCoords, nnz, _cIntArray (indexCoords), _cIntArray (indexVars), _cDoubleArray (coefs), _cDoubleArray (constants))
        )


#-------------------------------------------------------------------------------
#     ADDING COMPLEMENTARITY CONSTRAINTS
#-------------------------------------------------------------------------------

#---- KN_set_compcons
_knitro.KN_set_compcons.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_compcons.restype = ctypes.c_int
def KN_set_compcons (kc, ccTypes, indexComps1, indexComps2):
    if ccTypes is not None:
        nCC = len (ccTypes)
        if (indexComps1 is None or nCC != len (indexComps1)) and (indexComps2 is None or nCC != len (indexComps2)):
            raise ValueError ("Knitro-Python error: Arrays ccTypes, indexComps1 and indexComps2 have different sizes!")
        _checkRaise (
            "KN_set_compcons",
            _knitro.KN_set_compcons (kc, nCC, _cIntArray (ccTypes), _cIntArray (indexComps1), _cIntArray (indexComps2))
        )


#-------------------------------------------------------------------------------
#     CALLBACKS
#-------------------------------------------------------------------------------

#---- KN_add_eval_callback
_knitro.KN_add_eval_callback.argtypes = [KN_context_ptr, ctypes.c_bool, ctypes.c_int, ctypes.POINTER (ctypes.c_int), _KN_eval_callback, ctypes.POINTER (CB_context_ptr)]
_knitro.KN_add_eval_callback.restype = ctypes.c_int
_knitro.KN_add_eval_callback_all.argtypes = [KN_context_ptr, _KN_eval_callback, ctypes.POINTER (CB_context_ptr)]
_knitro.KN_add_eval_callback_all.restype = ctypes.c_int
_knitro.KN_add_eval_callback_one.argtypes = [KN_context_ptr, ctypes.c_int, _KN_eval_callback, ctypes.POINTER (CB_context_ptr)]
_knitro.KN_add_eval_callback_one.restype = ctypes.c_int
def KN_add_eval_callback (kc, evalObj = None, indexCons = None, funcCallback = None):
    '''Register evaluation callback

    The argument funcCallback is expected to be a Python method with the following
    prototype:
    def funcCallback (kc, evalRequest, evalResult, userParams)
    '''
    fnPtrWrapper = _KN_eval_callback_wrapper (kc, funcCallback)
    _registerCallback (kc, fnPtrWrapper)
    cb = CB_context_ptr ()
    if indexCons is None:
        if evalObj is None:
            _checkRaise ("KN_add_eval_callback_all", _knitro.KN_add_eval_callback_all (kc, fnPtrWrapper.c_fnPtr, ctypes.byref (cb)))
        elif evalObj:
            _checkRaise ("KN_add_eval_callback_one", _knitro.KN_add_eval_callback_one (kc, -1, fnPtrWrapper.c_fnPtr, ctypes.byref (cb)))
    else:
        try:
            nC = len (indexCons)
            if evalObj is None:
                evalObj = False
            _checkRaise ("KN_add_eval_callback", _knitro.KN_add_eval_callback (kc, evalObj, nC, _cIntArray (indexCons), fnPtrWrapper.c_fnPtr, ctypes.byref (cb)))
        except TypeError:
            if evalObj:
                raise ValueError ("Knitro-Python error: Argument evalObj cannot be True when setting callback for a single constraint!")
            _checkRaise ("KN_add_eval_callback_one", _knitro.KN_add_eval_callback_one (kc, indexCons, fnPtrWrapper.c_fnPtr, ctypes.byref (cb)))
    return cb

#---- KN_add_lsq_eval_callback
_knitro.KN_add_lsq_eval_callback.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), _KN_eval_callback, ctypes.POINTER (CB_context_ptr)]
_knitro.KN_add_lsq_eval_callback.restype = ctypes.c_int
_knitro.KN_add_lsq_eval_callback_all.argtypes = [KN_context_ptr, _KN_eval_callback, ctypes.POINTER (CB_context_ptr)]
_knitro.KN_add_lsq_eval_callback_all.restype = ctypes.c_int
_knitro.KN_add_lsq_eval_callback_one.argtypes = [KN_context_ptr, ctypes.c_int, _KN_eval_callback, ctypes.POINTER (CB_context_ptr)]
_knitro.KN_add_lsq_eval_callback_one.restype = ctypes.c_int
def KN_add_lsq_eval_callback (kc, indexRsds = None, rsdCallback = None):
    '''Register least-square evaluation callback

    The argument rsdCallback is expected to be a Python method with the following
    prototype:
    def rsdCallback (kc, evalRequest, evalResult, userParams)
    '''
    fnPtrWrapper = _KN_eval_callback_wrapper (kc, rsdCallback)
    _registerCallback (kc, fnPtrWrapper)
    cb = CB_context_ptr ()
    if indexRsds is None:
        _checkRaise ("KN_add_lsq_eval_callback_all", _knitro.KN_add_lsq_eval_callback_all (kc, fnPtrWrapper.c_fnPtr, ctypes.byref (cb)))
    else:
        try:
            nR = len (indexRsds)
            _checkRaise ("KN_add_lsq_eval_callback", _knitro.KN_add_lsq_eval_callback (kc, evalObj, nR, _cIntArray (indexRsds), fnPtrWrapper.c_fnPtr, ctypes.byref (cb)))
        except TypeError:
            _checkRaise ("KN_add_lsq_eval_callback_one", _knitro.KN_add_lsq_eval_callback_one (kc, indexRsd, fnPtrWrapper.c_fnPtr, ctypes.byref (cb)))
    return cb

#---- KN_set_cb_grad
_knitro.KN_set_cb_grad.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), KNLONG, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), _KN_eval_callback]
_knitro.KN_set_cb_grad.restype = ctypes.c_int
def KN_set_cb_grad (kc, cb, objGradIndexVars = None, jacIndexCons = None, jacIndexVars = None, gradCallback = None):
    nV = 0
    c_objGradIndexVars = None
    if objGradIndexVars is not None:
        try:
            nV = len (objGradIndexVars)
            c_objGradIndexVars = _cIntArray (objGradIndexVars)
        except TypeError:
            if objGradIndexVars < 0: # KN_DENSE
                nV = objGradIndexVars
            else:
                nV = 1
                c_objGradIndexVars = _cIntPointer (objGradIndexVars)
    nnz = 0
    c_jacIndexCons = None
    c_jacIndexVars = None

    # Check the sparsity pattern validity
    if not _valid_sparsity_pattern(jacIndexCons, jacIndexVars):
        raise ValueError ("Knitro-Python error: Arrays jacIndexCons and jacIndexVars have incompatible values!")

    if jacIndexCons is not None:
        try:
            nnz = len (jacIndexCons)
            if jacIndexVars is None or nnz != len (jacIndexVars):
                raise ValueError ("Knitro-Python error: Arrays jacIndexCons and jacIndexVars have different sizes!")
            c_jacIndexCons = _cIntArray (jacIndexCons)
            c_jacIndexVars = _cIntArray (jacIndexVars)
        except TypeError:
            if jacIndexCons < 0: # KN_DENSE_*
                nnz = jacIndexCons
            else:
                nnz = 1
                c_jacIndexCons = _cIntPointer (jacIndexCons)
                c_jacIndexVars = _cIntPointer (jacIndexVars)
    elif jacIndexVars is None:
        pass
    elif jacIndexVars < 0: # KN_DENSE_*:
        nnz = jacIndexVars
        c_jacIndexCons = None
        c_jacIndexVars = None
    else:
        raise ValueError ("Knitro-Python error: Arrays jacIndexCons and jacIndexVars have different sizes!")
    fnPtrWrapper = _KN_eval_callback_wrapper (kc, gradCallback)
    _registerCallback (kc, fnPtrWrapper)
    _checkRaise ("KN_set_cb_grad", _knitro.KN_set_cb_grad (kc, cb, nV, c_objGradIndexVars, nnz, c_jacIndexCons, c_jacIndexVars, fnPtrWrapper.c_fnPtr))

#---- KN_set_cb_hess
_knitro.KN_set_cb_hess.argtypes = [KN_context_ptr, CB_context_ptr, KNLONG, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), _KN_eval_callback]
_knitro.KN_set_cb_hess.restype = ctypes.c_int
def KN_set_cb_hess (kc, cb, hessIndexVars1 = None, hessIndexVars2 = None, hessCallback = None):
    fnPtrWrapper = _KN_eval_callback_wrapper (kc, hessCallback)
    _registerCallback (kc, fnPtrWrapper)
    nnz = 0

    # Check the sparsity pattern validity
    if not _valid_sparsity_pattern(hessIndexVars1, hessIndexVars2):
        raise ValueError ("Knitro-Python error: Arrays hessIndexVars1 and hessIndexVars2 have incompatible values!")

    if hessIndexVars1 is not None:
        try:
            nnz = len (hessIndexVars1)
            if hessIndexVars2 is None or nnz != len (hessIndexVars2):
                raise ValueError ("Knitro-Python error: Arrays hessIndexVars1 and hessIndexVars2 have different sizes!")
            _checkRaise ("KN_set_cb_hess", _knitro.KN_set_cb_hess (kc, cb, nnz, _cIntArray (hessIndexVars1), _cIntArray (hessIndexVars2), fnPtrWrapper.c_fnPtr))
        except TypeError:
            if hessIndexVars1 < 0: # KN_DENSE_*
                _checkRaise ("KN_set_cb_hess", _knitro.KN_set_cb_hess (kc, cb, hessIndexVars1, None, None, fnPtrWrapper.c_fnPtr))
            else:
                _checkRaise ("KN_set_cb_hess", _knitro.KN_set_cb_hess (kc, cb, 1, _cIntPointer (hessIndexVars1), _cIntPointer (hessIndexVars2), fnPtrWrapper.c_fnPtr))
    elif hessIndexVars2 is None:
        return
    elif hessIndexVars2 < 0: # KN_DENSE_*
        _checkRaise ("KN_set_cb_hess", _knitro.KN_set_cb_hess (kc, cb, hessIndexVars2, None, None, fnPtrWrapper.c_fnPtr))
    else:
        raise ValueError ("Knitro-Python error: Arrays hessIndexVars1 and hessIndexVars2 have different sizes!")

#---- KN_set_cb_rsd_jac
_knitro.KN_set_cb_rsd_jac.argtypes = [KN_context_ptr, CB_context_ptr, KNLONG, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), _KN_eval_callback]
_knitro.KN_set_cb_rsd_jac.restype = ctypes.c_int
def KN_set_cb_rsd_jac (kc, cb, jacIndexRsds = None, jacIndexVars = None, rsdJacCallback = None):
    fnPtrWrapper = _KN_eval_callback_wrapper (kc, rsdJacCallback)
    _registerCallback (kc, fnPtrWrapper)
    nnz = 0

    # Check the sparsity pattern validity
    if not _valid_sparsity_pattern(jacIndexRsds, jacIndexVars):
        raise ValueError ("Knitro-Python error: Arrays jacIndexRsds and jacIndexVars have incompatible values!")

    if jacIndexRsds is not None:
        try:
            nnz = len (jacIndexRsds)
            if jacIndexVars is None or nnz != len (jacIndexVars):
                raise ValueError ("Knitro-Python error: Arrays jacIndexRsds and jacIndexVars have different sizes!")
            _checkRaise ("KN_set_cb_rsd_jac", _knitro.KN_set_cb_rsd_jac (kc, cb, nnz, _cIntArray (jacIndexRsds), _cIntArray (jacIndexVars), fnPtrWrapper.c_fnPtr))
        except TypeError:
            if jacIndexRsds < 0: # KN_DENSE_*
                _checkRaise ("KN_set_cb_rsd_jac", _knitro.KN_set_cb_rsd_jac (kc, cb, jacIndexRsds, None, None, fnPtrWrapper.c_fnPtr))
            else:
                _checkRaise ("KN_set_cb_rsd_jac", _knitro.KN_set_cb_rsd_jac (kc, cb, 1, _cIntPointer (jacIndexRsds), _cIntPointer (jacIndexVars), fnPtrWrapper.c_fnPtr))
    elif jacIndexVars is None:
        return
    elif jacIndexVars < 0:
        _checkRaise ("KN_set_cb_rsd_jac", _knitro.KN_set_cb_rsd_jac (kc, cb, jacIndexVars, None, None, fnPtrWrapper.c_fnPtr))
    else:
        raise ValueError ("Knitro-Python error: Arrays jacIndexRsds and jacIndexVars have different sizes!")

#---- KN_set_cb_user_params
_knitro.KN_set_cb_user_params.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.c_void_p]
_knitro.KN_set_cb_user_params.restype = ctypes.c_int
def KN_set_cb_user_params (kc, cb, userParams):
    _registerUserParams (kc, userParams)
    # Knitro's internal KN_set_cb_user_params() function not called since it is unnecessary

#---- KN_set_cb_gradopt
_knitro.KN_set_cb_gradopt.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.c_int]
_knitro.KN_set_cb_gradopt.restype = ctypes.c_int
def KN_set_cb_gradopt (kc, cb, gradopt):
    _checkRaise ("KN_set_cb_gradopt", _knitro.KN_set_cb_gradopt (kc, cb, gradopt))

#---- KN_set_cb_relstepsizes
_knitro.KN_set_cb_relstepsizes.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_cb_relstepsizes.restype = ctypes.c_int
_knitro.KN_set_cb_relstepsizes_all.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_cb_relstepsizes_all.restype = ctypes.c_int
_knitro.KN_set_cb_relstepsize.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_cb_relstepsize.restype = ctypes.c_int
def KN_set_cb_relstepsizes (kc, cb, indexVars = None, xRelStepSizes = None):
    if indexVars is None:
        _checkRaise ("KN_set_cb_relstepsizes_all", _knitro.KN_set_cb_relstepsizes_all (kc, cb, _cDoubleArray (xRelStepSizes)))
    else:
        try:
            nV = len (indexVars)
            if xRelStepSizes is None or nV != len (xRelStepSizes):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xRelStepSizes have different sizes!")
            _checkRaise ("KN_set_cb_relstepsizes", _knitro.KN_set_cb_relstepsizes (kc, cb, nV, _cIntArray (indexVars), _cDoubleArray (xRelStepSizes)))
        except TypeError:
            _checkRaise ("KN_set_cb_relstepsize", _knitro.KN_set_cb_relstepsize (kc, cb, indexVars, xRelStepSizes))

#---- KN_get_cb_number_cons
_knitro.KN_get_cb_number_cons.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_cb_number_cons.restype = ctypes.c_int
def KN_get_cb_number_cons (kc, cb):
    c_nC = ctypes.c_int (0)
    _checkRaise ("KN_get_cb_number_cons", _knitro.KN_get_cb_number_cons (kc, cb, ctypes.byref (c_nC)))
    return c_nC.value

#---- KN_get_cb_number_rsds
_knitro.KN_get_cb_number_rsds.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_cb_number_rsds.restype = ctypes.c_int
def KN_get_cb_number_rsds (kc, cb):
    c_nR = ctypes.c_int (0)
    _checkRaise ("KN_get_cb_number_rsds", _knitro.KN_get_cb_number_rsds (kc, cb, ctypes.byref (c_nR)))
    return c_nR.value

#---- KN_get_cb_objgrad_nnz
_knitro.KN_get_cb_objgrad_nnz.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_cb_objgrad_nnz.restype = ctypes.c_int
def KN_get_cb_objgrad_nnz (kc, cb):
    c_nnz = ctypes.c_int (0)
    _checkRaise ("KN_get_cb_objgrad_nnz", _knitro.KN_get_cb_objgrad_nnz (kc, cb, ctypes.byref (c_nnz)))
    return c_nnz.value

#---- KN_get_cb_jacobian_nnz
_knitro.KN_get_cb_jacobian_nnz.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_cb_jacobian_nnz.restype = ctypes.c_int
def KN_get_cb_jacobian_nnz (kc, cb):
    c_nnz = ctypes.c_int (0)
    _checkRaise ("KN_get_cb_jacobian_nnz", _knitro.KN_get_cb_jacobian_nnz (kc, cb, ctypes.byref (c_nnz)))
    return c_nnz.value

#---- KN_get_cb_rsd_jacobian_nnz
_knitro.KN_get_cb_rsd_jacobian_nnz.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_cb_rsd_jacobian_nnz.restype = ctypes.c_int
def KN_get_cb_rsd_jacobian_nnz (kc, cb):
    c_nnz = ctypes.c_int (0)
    _checkRaise ("KN_get_cb_rsd_jacobian_nnz", _knitro.KN_get_cb_rsd_jacobian_nnz (kc, cb, ctypes.byref (c_nnz)))
    return c_nnz.value

#---- KN_get_cb_hessian_nnz
_knitro.KN_get_cb_hessian_nnz.argtypes = [KN_context_ptr, CB_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_cb_hessian_nnz.restype = ctypes.c_int
def KN_get_cb_hessian_nnz (kc, cb):
    c_nnz = ctypes.c_int (0)
    _checkRaise ("KN_get_cb_hessian_nnz", _knitro.KN_get_cb_hessian_nnz (kc, cb, ctypes.byref (c_nnz)))
    return c_nnz.value

#---- KN_set_newpt_callback
_knitro.KN_set_newpt_callback.argtypes = [KN_context_ptr, _KN_user_callback, ctypes.c_void_p]
_knitro.KN_set_newpt_callback.restype = ctypes.c_int
def KN_set_newpt_callback (kc, fnPtr, userParams = None):
    fnPtrWrapper = _KN_user_callback_wrapper (kc, fnPtr)
    _registerCallback (kc, fnPtrWrapper)
    _registerUserParams (kc, userParams)
    _checkRaise ("KN_set_newpt_callback", _knitro.KN_set_newpt_callback (kc, fnPtrWrapper.c_fnPtr, None))

#---- KN_set_mip_node_callback
_knitro.KN_set_mip_node_callback.argtypes = [KN_context_ptr, _KN_user_callback, ctypes.c_void_p]
_knitro.KN_set_mip_node_callback.restype = ctypes.c_int
def KN_set_mip_node_callback (kc, fnPtr, userParams = None):
    fnPtrWrapper = _KN_user_callback_wrapper (kc, fnPtr)
    _registerCallback (kc, fnPtrWrapper)
    _registerUserParams (kc, userParams)
    _checkRaise ("KN_set_mip_node_callback", _knitro.KN_set_mip_node_callback (kc, fnPtrWrapper.c_fnPtr, None))

#---- KN_set_ms_process_callback
_knitro.KN_set_ms_process_callback.argtypes = [KN_context_ptr, _KN_user_callback, ctypes.c_void_p]
_knitro.KN_set_ms_process_callback.restype = ctypes.c_int
def KN_set_ms_process_callback (kc, fnPtr, userParams = None):
    fnPtrWrapper = _KN_user_callback_wrapper (kc, fnPtr)
    _registerCallback (kc, fnPtrWrapper)
    _registerUserParams (kc, userParams)
    _checkRaise ("KN_set_ms_process_callback", _knitro.KN_set_ms_process_callback (kc, fnPtrWrapper.c_fnPtr, None))

#---- KN_set_ms_initpt_callback
_knitro.KN_set_ms_initpt_callback.argtypes = [KN_context_ptr, _KN_ms_initpt_callback, ctypes.c_void_p]
_knitro.KN_set_ms_initpt_callback.restype = ctypes.c_int
def KN_set_ms_initpt_callback (kc, fnPtr, userParams = None):
    fnPtrWrapper = _KN_ms_initpt_callback_wrapper (kc, fnPtr)
    _registerCallback (kc, fnPtrWrapper)
    _registerUserParams (kc, userParams)
    _checkRaise ("KN_set_ms_initpt_callback", _knitro.KN_set_ms_initpt_callback (kc, fnPtrWrapper.c_fnPtr, None))

#---- KN_set_puts_callback
_knitro.KN_set_puts_callback.argtypes = [KN_context_ptr, _KN_puts, ctypes.c_void_p]
_knitro.KN_set_puts_callback.restype = ctypes.c_int
def KN_set_puts_callback (kc, fnPtr, userParams = None):
    fnPtrWrapper = _KN_puts_wrapper (kc, fnPtr)
    _registerCallback (kc, fnPtrWrapper)
    _registerUserParams (kc, userParams)
    _checkRaise ("KN_set_puts_callback", _knitro.KN_set_puts_callback (kc, fnPtrWrapper.c_fnPtr, None))


def _valid_sparsity_pattern(index1, index2):
    # (index1, index2) defines the sparsity pattern for jacobian of constraints/residuals or hessian
    # Only authorized combinations of (index1, index2) are:
    # (-int, -int), (None, -int), (-int, None): dense
    # (+int, +int), (list, list), (nparray, nparray): sparse
    # (None, None): Knitro approximation

    validSparsityPattern = ((type(index1) is type(index2)) or
                            (index1 is None and isinstance(index2, int) and index2 < 0) or # (None, -int)
                            (index2 is None and isinstance(index1, int) and index1 < 0))   # (-int, None)

    # Exclude cases (-int, +int) and (+int, -int)
    validIntegerPattern = (not isinstance(index1, int) or
                           not isinstance(index2, int) or
                           (index1 < 0) == (index2 < 0))

    return (validSparsityPattern and validIntegerPattern)

#-------------------------------------------------------------------------------
#     ALGORITHMIC FEATURES
#-------------------------------------------------------------------------------

#---- KN_set_var_feastols
_knitro.KN_set_var_feastols.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_feastols.restype = ctypes.c_int
_knitro.KN_set_var_feastols_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_feastols_all.restype = ctypes.c_int
_knitro.KN_set_var_feastol.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_var_feastol.restype = ctypes.c_int
def KN_set_var_feastols (kc, indexVars = None, xFeasTols = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_feastols_all", _knitro.KN_set_var_feastols_all (kc, _cDoubleArray (xFeasTols)))
    else:
        try:
            nV = len (indexVars)
            if xFeasTols is None or nV != len (xFeasTols):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xFeasTols have different sizes!")
            _checkRaise ("KN_set_var_feastols", _knitro.KN_set_var_feastols (kc, nV, _cIntArray (indexVars), _cDoubleArray (xFeasTols)))
        except TypeError:
            _checkRaise ("KN_set_var_feastol", _knitro.KN_set_var_feastol (kc, indexVars, xFeasTols))

#---- KN_set_con_feastols
_knitro.KN_set_con_feastols.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_feastols.restype = ctypes.c_int
_knitro.KN_set_con_feastols_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_feastols_all.restype = ctypes.c_int
_knitro.KN_set_con_feastol.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_con_feastol.restype = ctypes.c_int
def KN_set_con_feastols (kc, indexCons = None, cFeasTols = None):
    if indexCons is None:
        _checkRaise ("KN_set_con_feastols_all", _knitro.KN_set_con_feastols_all (kc, _cDoubleArray (cFeasTols)))
    else:
        try:
            nV = len (indexCons)
            if cFeasTols is None or nV != len (cFeasTols):
                raise ValueError ("Knitro-Python error: Arrays indexCons and cFeasTols have different sizes!")
            _checkRaise ("KN_set_con_feastols", _knitro.KN_set_con_feastols (kc, nV, _cIntArray (indexCons), _cDoubleArray (cFeasTols)))
        except TypeError:
            _checkRaise ("KN_set_con_feastol", _knitro.KN_set_con_feastol (kc, indexCons, cFeasTols))

#---- KN_set_compcon_feastols
_knitro.KN_set_compcon_feastols.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_compcon_feastols.restype = ctypes.c_int
_knitro.KN_set_compcon_feastols_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_compcon_feastols_all.restype = ctypes.c_int
_knitro.KN_set_compcon_feastol.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_compcon_feastol.restype = ctypes.c_int
def KN_set_compcon_feastols (kc, indexCompCons = None, ccFeasTols = None):
    if indexCompCons is None:
        _checkRaise ("KN_set_compcon_feastols_all", _knitro.KN_set_compcon_feastols_all (kc, _cDoubleArray (ccFeasTols)))
    else:
        try:
            nV = len (indexCompCons)
            if ccFeasTols is None or nV != len (ccFeasTols):
                raise ValueError ("Knitro-Python error: Arrays indexCompCons and ccFeasTols have different sizes!")
            _checkRaise ("KN_set_compcon_feastols", _knitro.KN_set_compcon_feastols (kc, nV, _cIntArray (indexCompCons), _cDoubleArray (ccFeasTols)))
        except TypeError:
            _checkRaise ("KN_set_compcon_feastol", _knitro.KN_set_compcon_feastol (kc, indexCompCons, ccFeasTols))

#---- KN_set_var_scalings
_knitro.KN_set_var_scalings.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_scalings.restype = ctypes.c_int
_knitro.KN_set_var_scalings_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_var_scalings_all.restype = ctypes.c_int
_knitro.KN_set_var_scaling.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_knitro.KN_set_var_scaling.restype = ctypes.c_int
def KN_set_var_scalings (kc, indexVars = None, xScaleFactors = None, xScaleCenters = None):
    if indexVars is None:
        nV = len (xScaleFactors)
        if xScaleCenters is None or nV != len (xScaleCenters):
            raise ValueError ("Knitro-Python error: Arrays indexVars, xScaleFactors and xScaleCenters have different sizes!")
        _checkRaise ("KN_set_var_scalings_all", _knitro.KN_set_var_scalings_all (kc, _cDoubleArray (xScaleFactors), _cDoubleArray (xScaleCenters)))
    else:
        try:
            nV = len (indexVars)
            if xScaleFactors is None or nV != len (xScaleFactors) or xScaleCenters is None or nV != len (xScaleCenters):
                raise ValueError ("Knitro-Python error: Arrays indexVars, xScaleFactors and xScaleCenters have different sizes!")
            _checkRaise ("KN_set_var_scalings", _knitro.KN_set_var_scalings (kc, nV, _cIntArray (indexVars), _cDoubleArray (xScaleFactors), _cDoubleArray (xScaleCenters)))
        except TypeError:
            _checkRaise ("KN_set_var_scaling", _knitro.KN_set_var_scaling (kc, indexVars, xScaleFactors, xScaleCenters))

#---- KN_set_con_scalings
_knitro.KN_set_con_scalings.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_scalings.restype = ctypes.c_int
_knitro.KN_set_con_scalings_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_con_scalings_all.restype = ctypes.c_int
_knitro.KN_set_con_scaling.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_con_scaling.restype = ctypes.c_int
def KN_set_con_scalings (kc, indexCons = None, cScaleFactors = None):
    if indexCons is None:
        _checkRaise ("KN_set_con_scalings_all", _knitro.KN_set_con_scalings_all (kc, _cDoubleArray (cScaleFactors)))
    else:
        try:
            nV = len (indexCons)
            if cScaleFactors is None or nV != len (cScaleFactors):
                raise ValueError ("Knitro-Python error: Arrays indexCons and cScaleFactors have different sizes!")
            _checkRaise ("KN_set_con_scalings", _knitro.KN_set_con_scalings (kc, nV, _cIntArray (indexCons), _cDoubleArray (cScaleFactors)))
        except TypeError:
            _checkRaise ("KN_set_con_scaling", _knitro.KN_set_con_scaling (kc, indexCons, cScaleFactors))

#---- KN_set_compcon_scalings
_knitro.KN_set_compcon_scalings.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_compcon_scalings.restype = ctypes.c_int
_knitro.KN_set_compcon_scalings_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_set_compcon_scalings_all.restype = ctypes.c_int
_knitro.KN_set_compcon_scaling.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_double]
_knitro.KN_set_compcon_scaling.restype = ctypes.c_int
def KN_set_compcon_scalings (kc, indexCompCons = None, ccScaleFactors = None):
    if indexCompCons is None:
        _checkRaise ("KN_set_compcon_scalings_all", _knitro.KN_set_compcon_scalings_all (kc, _cDoubleArray (ccScaleFactors)))
    else:
        try:
            nV = len (indexCompCons)
            if ccScaleFactors is None or nV != len (ccScaleFactors):
                raise ValueError ("Knitro-Python error: Arrays indexCompCons and ccScaleFactors have different sizes!")
            _checkRaise ("KN_set_compcon_scalings", _knitro.KN_set_compcon_scalings (kc, nV, _cIntArray (indexCompCons), _cDoubleArray (ccScaleFactors)))
        except TypeError:
            _checkRaise ("KN_set_compcon_scaling", _knitro.KN_set_compcon_scaling (kc, indexCompCons, ccScaleFactors))

#---- KN_set_obj_scaling
_knitro.KN_set_obj_scaling.argtypes = [KN_context_ptr, ctypes.c_double]
_knitro.KN_set_obj_scaling.restype = ctypes.c_int
def KN_set_obj_scaling (kc, objScaleFactor):
    _checkRaise ("KN_set_obj_scaling", _knitro.KN_set_obj_scaling (kc, objScaleFactor))

#---- KN_set_var_names
_knitro.KN_set_var_names.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_char_p)]
_knitro.KN_set_var_names.restype = ctypes.c_int
_knitro.KN_set_var_names_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_char_p)]
_knitro.KN_set_var_names_all.restype = ctypes.c_int
_knitro.KN_set_var_name.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_char_p]
_knitro.KN_set_var_name.restype = ctypes.c_int
def KN_set_var_names (kc, indexVars = None, xNames = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_names_all", _knitro.KN_set_var_names_all (kc, _cStringArray(xNames)))
    else:
        try:
            nV = len (indexVars)
            _checkRaise ("KN_set_var_names", _knitro.KN_set_var_names (kc, nV, _cIntArray (indexVars), _cStringArray(xNames)))
        except TypeError:
            _checkRaise ("KN_set_var_name", _knitro.KN_set_var_name (kc, indexVars, xNames.encode ('UTF-8')))

#---- KN_get_var_names
_knitro.KN_get_var_name.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
_knitro.KN_get_var_name.restype = ctypes.c_int
def KN_get_var_names(kc, indexVars=None, max_length=1024):
    # By default, return all names
    if indexVars is None:
        return [KN_get_var_names(kc, indexVars=i, max_length=max_length)
                for i in range(KN_get_number_vars(kc))]
    elif isiterable(indexVars):
        return [KN_get_var_names(kc, indexVars=i, max_length=max_length)
                for i in indexVars]
    elif isinstance(indexVars, int):
        c_name = ctypes.c_char_p((" " * max_length).encode('UTF-8'))
        _checkRaise ("KN_get_var_name", _knitro.KN_get_var_name(kc, indexVars, max_length, c_name))
        return c_name.value.decode('UTF-8')
    else:
        raise TypeError("Error in KN_get_var_names: variable indexVars sould be"
                        " None, int or list")

#---- KN_set_con_names
_knitro.KN_set_con_names.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_char_p)]
_knitro.KN_set_con_names.restype = ctypes.c_int
_knitro.KN_set_con_names_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_char_p)]
_knitro.KN_set_con_names_all.restype = ctypes.c_int
_knitro.KN_set_con_name.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_char_p]
_knitro.KN_set_con_name.restype = ctypes.c_int
def KN_set_con_names (kc, indexCons = None, cNames = None):
    if indexCons is None:
        _checkRaise ("KN_set_con_names_all", _knitro.KN_set_con_names_all (kc, _cStringArray(cNames)))
    else:
        try:
            nC = len (indexCons)
            _checkRaise ("KN_set_con_names", _knitro.KN_set_con_names (kc, nC, _cIntArray (indexCons), _cStringArray(cNames)))
        except TypeError:
            _checkRaise ("KN_set_con_name", _knitro.KN_set_con_name (kc, indexCons, cNames.encode ('UTF-8')))

#---- KN_get_con_names
_knitro.KN_get_con_name.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
_knitro.KN_get_con_name.restype = ctypes.c_int
def KN_get_con_names(kc, indexCons=None, max_length=1024):
    # By default, return all names
    if indexCons is None:
        return [KN_get_con_names(kc, indexCons=i, max_length=max_length)
                for i in range(KN_get_number_cons(kc))]
    elif isiterable(indexCons):
        return [KN_get_con_names(kc, indexCons=i, max_length=max_length)
                for i in indexCons]
    elif isinstance(indexCons, int):
        c_name = ctypes.c_char_p((" " * max_length).encode('UTF-8'))
        _checkRaise ("KN_get_con_name", _knitro.KN_get_con_name(kc, indexCons, max_length, c_name))
        return c_name.value.decode('UTF-8')
    else:
        raise TypeError("Error in KN_get_con_names: indexCons sould be"
                        " None, int or list")

#---- KN_set_compcon_names
_knitro.KN_set_compcon_names.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_char_p)]
_knitro.KN_set_compcon_names.restype = ctypes.c_int
_knitro.KN_set_compcon_names_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_char_p)]
_knitro.KN_set_compcon_names_all.restype = ctypes.c_int
_knitro.KN_set_compcon_name.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_char_p]
_knitro.KN_set_compcon_name.restype = ctypes.c_int
def KN_set_compcon_names (kc, indexCompCons = None, cNames = None):
    if indexCompCons is None:
        _checkRaise ("KN_set_compcon_names_all", _knitro.KN_set_compcon_names_all (kc, _cStringArray(cNames)))
    else:
        try:
            nCC = len (indexCompCons)
            _checkRaise ("KN_set_compcon_names", _knitro.KN_set_compcon_names (kc, nCC, _cIntArray (indexCompCons), _cStringArray(cNames)))
        except TypeError:
            _checkRaise ("KN_set_compcon_name", _knitro.KN_set_compcon_name (kc, indexCompCons, cNames.encode ('UTF-8')))

#---- KN_set_obj_name
_knitro.KN_set_obj_name.argtypes = [KN_context_ptr, ctypes.c_char_p]
_knitro.KN_set_obj_name.restype = ctypes.c_int
def KN_set_obj_name (kc, objName):
    _checkRaise ("KN_set_obj_name", _knitro.KN_set_obj_name (kc, objName.encode ('UTF-8')))

#---- KN_set_var_honorbnds
_knitro.KN_set_var_honorbnds.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_var_honorbnds.restype = ctypes.c_int
_knitro.KN_set_var_honorbnds_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_var_honorbnds_all.restype = ctypes.c_int
_knitro.KN_set_var_honorbnd.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int]
_knitro.KN_set_var_honorbnd.restype = ctypes.c_int
def KN_set_var_honorbnds (kc, indexVars = None, xHonorBnds = None):
    if indexVars is None:
        _checkRaise ("KN_set_var_honorbnds_all", _knitro.KN_set_var_honorbnds_all (kc, _cIntArray (xHonorBnds)))
    else:
        try:
            nV = len (indexVars)
            if xHonorBnds is None or nV != len (xHonorBnds):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xHonorBnds have different sizes!")
            _checkRaise ("KN_set_var_honorbnds", _knitro.KN_set_var_honorbnds (kc, nV, _cIntArray (indexVars), _cIntArray (xHonorBnds)))
        except TypeError:
            _checkRaise ("KN_set_var_honorbnd", _knitro.KN_set_var_honorbnd (kc, indexVars, xHonorBnds))

#---- KN_set_mip_branching_priorities
_knitro.KN_set_mip_branching_priorities.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_mip_branching_priorities.restype = ctypes.c_int
_knitro.KN_set_mip_branching_priorities_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_mip_branching_priorities_all.restype = ctypes.c_int
_knitro.KN_set_mip_branching_priority.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int]
_knitro.KN_set_mip_branching_priority.restype = ctypes.c_int
def KN_set_mip_branching_priorities (kc, indexVars = None, xPriorities = None):
    if indexVars is None:
        _checkRaise ("KN_set_mip_branching_priorities_all", _knitro.KN_set_mip_branching_priorities_all (kc, _cIntArray (xPriorities)))
    else:
        try:
            nV = len (indexVars)
            if xPriorities is None or nV != len (xPriorities):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xPriorities have different sizes!")
            _checkRaise ("KN_set_mip_branching_priorities", _knitro.KN_set_mip_branching_priorities (kc, nV, _cIntArray (indexVars), _cIntArray (xPriorities)))
        except TypeError:
            _checkRaise ("KN_set_mip_branching_priority", _knitro.KN_set_mip_branching_priority (kc, indexVars, xPriorities))

#---- KN_set_mip_intvar_strategies
_knitro.KN_set_mip_intvar_strategies.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_mip_intvar_strategies.restype = ctypes.c_int
_knitro.KN_set_mip_intvar_strategies_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_set_mip_intvar_strategies_all.restype = ctypes.c_int
_knitro.KN_set_mip_intvar_strategy.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.c_int]
_knitro.KN_set_mip_intvar_strategy.restype = ctypes.c_int
def KN_set_mip_intvar_strategies (kc, indexVars = None, xStrategies = None):
    if indexVars is None:
        _checkRaise ("KN_set_mip_intvar_strategies_all", _knitro.KN_set_mip_intvar_strategies_all (kc, _cIntArray (xStrategies)))
    else:
        try:
            nV = len (indexVars)
            if xStrategies is None or nV != len (xStrategies):
                raise ValueError ("Knitro-Python error: Arrays indexVars and xStrategies have different sizes!")
            _checkRaise ("KN_set_mip_intvar_strategies", _knitro.KN_set_mip_intvar_strategies (kc, nV, _cIntArray (indexVars), _cIntArray (xStrategies)))
        except TypeError:
            _checkRaise ("KN_set_mip_intvar_strategy", _knitro.KN_set_mip_intvar_strategy (kc, indexVars, xStrategies))


#-------------------------------------------------------------------------------
#     SOLVING
#-------------------------------------------------------------------------------

#---- KN_solve
_knitro.KN_solve.argtypes = [KN_context_ptr]
_knitro.KN_solve.restype = ctypes.c_int
def KN_solve (kc):
    # For KN_solve, we do not raise an error when the return status is not 0
    return _knitro.KN_solve (kc)


#-------------------------------------------------------------------------------
#     READING MODEL/SOLUTION PROPERTIES
#-------------------------------------------------------------------------------

#---- KN_get_number_vars
_knitro.KN_get_number_vars.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_number_vars.restype = ctypes.c_int
def KN_get_number_vars (kc):
    c_nV = ctypes.c_int (0)
    _checkRaise ("KN_get_number_vars", _knitro.KN_get_number_vars (kc, ctypes.byref (c_nV)))
    return c_nV.value

#---- KN_get_number_cons
_knitro.KN_get_number_cons.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_number_cons.restype = ctypes.c_int
def KN_get_number_cons (kc):
    c_nC = ctypes.c_int (0)
    _checkRaise ("KN_get_number_cons", _knitro.KN_get_number_cons (kc, ctypes.byref (c_nC)))
    return c_nC.value

#---- KN_get_number_rsds
_knitro.KN_get_number_rsds.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_number_rsds.restype = ctypes.c_int
def KN_get_number_rsds (kc):
    c_nR = ctypes.c_int (0)
    _checkRaise ("KN_get_number_rsds", _knitro.KN_get_number_rsds (kc, ctypes.byref (c_nR)))
    return c_nR.value

#---- KN_get_number_FC_evals
_knitro.KN_get_number_FC_evals.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_number_FC_evals.restype = ctypes.c_int
def KN_get_number_FC_evals (kc):
    c_numFCevals = ctypes.c_int (0)
    _checkRaise ("KN_get_number_FC_evals", _knitro.KN_get_number_FC_evals (kc, ctypes.byref (c_numFCevals)))
    return c_numFCevals.value

#---- KN_get_number_GA_evals
_knitro.KN_get_number_GA_evals.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_number_GA_evals.restype = ctypes.c_int
def KN_get_number_GA_evals (kc):
    c_numGAevals = ctypes.c_int (0)
    _checkRaise ("KN_get_number_GA_evals", _knitro.KN_get_number_GA_evals (kc, ctypes.byref (c_numGAevals)))
    return c_numGAevals.value

#---- KN_get_number_H_evals
_knitro.KN_get_number_H_evals.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_number_H_evals.restype = ctypes.c_int
def KN_get_number_H_evals (kc):
    c_numHevals = ctypes.c_int (0)
    _checkRaise ("KN_get_number_H_evals", _knitro.KN_get_number_H_evals (kc, ctypes.byref (c_numHevals)))
    return c_numHevals.value

#---- KN_get_number_HV_evals
_knitro.KN_get_number_HV_evals.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_number_HV_evals.restype = ctypes.c_int
def KN_get_number_HV_evals (kc):
    c_numHVevals = ctypes.c_int (0)
    _checkRaise ("KN_get_number_HV_evals", _knitro.KN_get_number_HV_evals (kc, ctypes.byref (c_numHVevals)))
    return c_numHVevals.value

#---- KN_get_solution
_knitro.KN_get_solution.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double), ctypes.POINTER (ctypes.c_double), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_solution.restype = ctypes.c_int
def KN_get_solution (kc):
    nV = KN_get_number_vars (kc)
    nC = KN_get_number_cons (kc)
    c_status = ctypes.c_int (0)
    c_objective = ctypes.c_double (0)
    c_x = (ctypes.c_double * nV) ()
    c_lambda = (ctypes.c_double * (nC+nV)) ()
    _checkRaise ("KN_get_solution", _knitro.KN_get_solution (kc, ctypes.byref (c_status), ctypes.byref (c_objective), c_x, c_lambda))
    return c_status.value, c_objective.value, _userArray (nV, c_x), _userArray (nC+nV, c_lambda)

#---- KN_get_obj_value
_knitro.KN_get_obj_value.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_obj_value.restype = ctypes.c_int
def KN_get_obj_value (kc):
    c_obj = ctypes.c_double (0)
    _checkRaise ("KN_get_obj_value", _knitro.KN_get_obj_value (kc, ctypes.byref (c_obj)))
    return c_obj.value

#---- KN_get_obj_type
_knitro.KN_get_obj_type.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_obj_type.restype = ctypes.c_int
def KN_get_obj_type (kc):
    c_objType = ctypes.c_int (0)
    _checkRaise ("KN_get_obj_type", _knitro.KN_get_obj_type (kc, ctypes.byref (c_objType)))
    return c_objType.value

#---- KN_get_con_values
_knitro.KN_get_con_values.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_con_values.restype = ctypes.c_int
_knitro.KN_get_con_values_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_con_values_all.restype = ctypes.c_int
_knitro.KN_get_con_value.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_con_value.restype = ctypes.c_int
def KN_get_con_values (kc, indexCons = None):
    if indexCons is None:
        nC = KN_get_number_cons (kc)
        c_c = (ctypes.c_double * nC) ()
        _checkRaise ("KN_get_con_values_all", _knitro.KN_get_con_values_all (kc, c_c))
        return _userArray (nC, c_c)
    else:
        try:
            nC = len (indexCons)
            c_c = (ctypes.c_double * nC) ()
            _checkRaise ("KN_get_con_values", _knitro.KN_get_con_values (kc, _cIntArray (indexCons), c_c))
            return _userArray (nC, c_c)
        except TypeError:
            c_c = ctypes.c_double (0)
            _checkRaise ("KN_get_con_value", _knitro.KN_get_con_value (kc, indexCons, ctypes.byref (c_c)))
            return c_c.value

#---- KN_get_con_types
_knitro.KN_get_con_types.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_con_types.restype = ctypes.c_int
_knitro.KN_get_con_types_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_con_types_all.restype = ctypes.c_int
_knitro.KN_get_con_type.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_con_type.restype = ctypes.c_int
def KN_get_con_types (kc, indexCons = None):
    if indexCons is None:
        nC = KN_get_number_cons (kc)
        c_cTypeType = (ctypes.c_int * nC) ()
        _checkRaise ("KN_get_con_types_all", _knitro.KN_get_con_types_all (kc, c_cType))
        return _userArray (nC, c_cType)
    else:
        try:
            nC = len (indexCons)
            c_cType = (ctypes.c_int * nC) ()
            _checkRaise ("KN_get_con_types", _knitro.KN_get_con_types (kc, _cIntArray (indexCons), c_cType))
            return _userArray (nC, c_cType)
        except TypeError:
            c_cType = ctypes.c_int (0)
            _checkRaise ("KN_get_con_type", _knitro.KN_get_con_type (kc, indexCons, ctypes.byref (c_cType)))
            return c_cType.value

#---- KN_get_rsd_values
_knitro.KN_get_rsd_values.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_rsd_values.restype = ctypes.c_int
_knitro.KN_get_rsd_values_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_rsd_values_all.restype = ctypes.c_int
_knitro.KN_get_rsd_value.argtypes = [KN_context_ptr, ctypes.c_int, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_rsd_value.restype = ctypes.c_int
def KN_get_rsd_values (kc, indexRsds = None):
    if indexRsds is None:
        nR = KN_get_number_cons (kc)
        c_r = (ctypes.c_double * nR) ()
        _checkRaise ("KN_get_rsd_values_all", _knitro.KN_get_rsd_values_all (kc, c_r))
        return _userArray (nR, c_r)
    else:
        try:
            nR = len (indexRsds)
            c_r = (ctypes.c_double * nR) ()
            _checkRaise ("KN_get_rsd_values", _knitro.KN_get_rsd_values (kc, _cIntArray (indexRsds), c_r))
            return _userArray (nR, c_r)
        except TypeError:
            c_r = ctypes.c_double (0)
            _checkRaise ("KN_get_rsd_value", _knitro.KN_get_rsd_value (kc, indexRsds, ctypes.byref (c_r)))
            return c_r.value


#-------------------------------------------------------------------------------
#     SOLUTION PROPERTIES FOR CONTINUOUS PROBLEMS ONLY
#-------------------------------------------------------------------------------

#--- KN_get_solve_time_cpu
_knitro.KN_get_solve_time_cpu.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_solve_time_cpu.restype = ctypes.c_int
def KN_get_solve_time_cpu(kc):
    c_solveTimeCPU = ctypes.c_double(0)
    _checkRaise ("KN_get_solve_time_cpu", _knitro.KN_get_solve_time_cpu (kc, ctypes.byref (c_solveTimeCPU)))
    return c_solveTimeCPU.value

#--- KN_get_solve_time_real
_knitro.KN_get_solve_time_real.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_solve_time_real.restype = ctypes.c_int
def KN_get_solve_time_real(kc):
    c_solveTimeReal = ctypes.c_double(0)
    _checkRaise ("KN_get_solve_time_real", _knitro.KN_get_solve_time_real (kc, ctypes.byref (c_solveTimeReal)))
    return c_solveTimeReal.value

#---- KN_get_number_iters
_knitro.KN_get_number_iters.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_number_iters.restype = ctypes.c_int
def KN_get_number_iters (kc):
    c_numIters = ctypes.c_int (0)
    _checkRaise ("KN_get_number_iters", _knitro.KN_get_number_iters (kc, ctypes.byref (c_numIters)))
    return c_numIters.value

#---- KN_get_number_cg_iters
_knitro.KN_get_number_cg_iters.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_number_cg_iters.restype = ctypes.c_int
def KN_get_number_cg_iters (kc):
    c_numCGiters = ctypes.c_int (0)
    _checkRaise ("KN_get_number_cg_iters", _knitro.KN_get_number_cg_iters (kc, ctypes.byref (c_numCGiters)))
    return c_numCGiters.value

#---- KN_get_abs_feas_error
_knitro.KN_get_abs_feas_error.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_abs_feas_error.restype = ctypes.c_int
def KN_get_abs_feas_error (kc):
    c_absFeasError = ctypes.c_double (0)
    _checkRaise ("KN_get_abs_feas_error", _knitro.KN_get_abs_feas_error (kc, ctypes.byref (c_absFeasError)))
    return c_absFeasError.value

#---- KN_get_rel_feas_error
_knitro.KN_get_rel_feas_error.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_rel_feas_error.restype = ctypes.c_int
def KN_get_rel_feas_error (kc):
    c_relFeasError = ctypes.c_double (0)
    _checkRaise ("KN_get_rel_feas_error", _knitro.KN_get_rel_feas_error (kc, ctypes.byref (c_relFeasError)))
    return c_relFeasError.value

#---- KN_get_abs_opt_error
_knitro.KN_get_abs_opt_error.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_abs_opt_error.restype = ctypes.c_int
def KN_get_abs_opt_error (kc):
    c_absOptError = ctypes.c_double (0)
    _checkRaise ("KN_get_abs_opt_error", _knitro.KN_get_abs_opt_error (kc, ctypes.byref (c_absOptError)))
    return c_absOptError.value

#---- KN_get_rel_opt_error
_knitro.KN_get_rel_opt_error.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_rel_opt_error.restype = ctypes.c_int
def KN_get_rel_opt_error (kc):
    c_relOptError = ctypes.c_double (0)
    _checkRaise ("KN_get_rel_opt_error", _knitro.KN_get_rel_opt_error (kc, ctypes.byref (c_relOptError)))
    return c_relOptError.value

#---- KN_get_objgrad_values
_knitro.KN_get_objgrad_nnz.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_objgrad_nnz.restype = ctypes.c_int
_knitro.KN_get_objgrad_values.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_objgrad_values.restype = ctypes.c_int
def KN_get_objgrad_values (kc):
    c_nnz = ctypes.c_int (0)
    _checkRaise ("KN_get_objgrad_nnz", _knitro.KN_get_objgrad_nnz (kc, ctypes.byref (c_nnz)))
    nnz = c_nnz.value
    c_indexVars = (ctypes.c_int * nnz) ()
    c_objGrad = (ctypes.c_double * nnz) ()
    _checkRaise ("KN_get_objgrad_values", _knitro.KN_get_objgrad_values (kc, c_indexVars, c_objGrad))
    return _userArray (nnz, c_indexVars), _userArray (nnz, c_objGrad)

#---- KN_get_objgrad_values_all
_knitro.KN_get_objgrad_values_all.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_objgrad_values_all.restype = ctypes.c_int
def KN_get_objgrad_values_all (kc):
    nV = KN_get_number_vars (kc)
    c_objGrad = (ctypes.c_double * nV) ()
    _checkRaise ("KN_get_objgrad_values_all", _knitro.KN_get_objgrad_values_all (kc, c_objGrad))
    return _userArray (nV, c_objGrad)

#---- KN_get_jacobian_values
_knitro.KN_get_jacobian_nnz.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_jacobian_nnz.restype = ctypes.c_int
_knitro.KN_get_jacobian_values.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_jacobian_values.restype = ctypes.c_int
def KN_get_jacobian_values (kc):
    c_nnz = ctypes.c_int (0)
    _checkRaise ("KN_get_jacobian_nnz", _knitro.KN_get_jacobian_nnz (kc, ctypes.byref (c_nnz)))
    nnz = c_nnz.value
    c_indexCons = (ctypes.c_int * nnz) ()
    c_indexVars = (ctypes.c_int * nnz) ()
    c_jac = (ctypes.c_double * nnz) ()
    _checkRaise ("KN_get_jacobian_values", _knitro.KN_get_jacobian_values (kc, c_indexCons, c_indexVars, c_jac))
    return _userArray (nnz, c_indexCons), _userArray (nnz, c_indexVars), _userArray (nnz, c_jac)

#---- KN_get_rsd_jacobian_values
_knitro.KN_get_rsd_jacobian_nnz.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_rsd_jacobian_nnz.restype = ctypes.c_int
_knitro.KN_get_rsd_jacobian_values.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_rsd_jacobian_values.restype = ctypes.c_int
def KN_get_rsd_jacobian_values (kc):
    c_nnz = ctypes.c_int (0)
    _checkRaise ("KN_get_rsd_jacobian_nnz", _knitro.KN_get_rsd_jacobian_nnz (kc, ctypes.byref (c_nnz)))
    nnz = c_nnz.value
    c_indexRsds = (ctypes.c_int * nnz) ()
    c_indexVars = (ctypes.c_int * nnz) ()
    c_rsdJac = (ctypes.c_double * nnz) ()
    _checkRaise ("KN_get_rsd_jacobian_values", _knitro.KN_get_rsd_jacobian_values (kc, c_indexRsds, c_indexVars, c_rsdJac))
    return _userArray (nnz, c_indexRsds), _userArray (nnz, c_indexVars), _userArray (nnz, c_rsdJac)

#---- KN_get_hessian_values
_knitro.KN_get_hessian_nnz.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_hessian_nnz.restype = ctypes.c_int
_knitro.KN_get_hessian_values.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_int), ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_hessian_values.restype = ctypes.c_int
def KN_get_hessian_values (kc):
    c_nnz = ctypes.c_int (0)
    _checkRaise ("KN_get_hessian_nnz", _knitro.KN_get_hessian_nnz (kc, ctypes.byref (c_nnz)))
    nnz = c_nnz.value
    c_indexVars1 = (ctypes.c_int * nnz) ()
    c_indexVars2 = (ctypes.c_int * nnz) ()
    c_hess = (ctypes.c_double * nnz) ()
    _checkRaise ("KN_get_hessian_values", _knitro.KN_get_hessian_values (kc, c_indexVars1, c_indexVars2, c_hess))
    return _userArray (nnz, c_indexVars1), _userArray (nnz, c_indexVars2), _userArray (nnz, c_hess)


#-------------------------------------------------------------------------------
#     SOLUTION PROPERTIES FOR MIP PROBLEMS ONLY
#-------------------------------------------------------------------------------

#---- KN_get_mip_number_nodes
_knitro.KN_get_mip_number_nodes.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_mip_number_nodes.restype = ctypes.c_int
def KN_get_mip_number_nodes (kc):
    c_numNodes = ctypes.c_int (0)
    _checkRaise ("KN_get_mip_number_nodes", _knitro.KN_get_mip_number_nodes (kc, ctypes.byref (c_numNodes)))
    return c_numNodes.value

#---- KN_get_mip_number_solves
_knitro.KN_get_mip_number_solves.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_int)]
_knitro.KN_get_mip_number_solves.restype = ctypes.c_int
def KN_get_mip_number_solves (kc):
    c_numSolves = ctypes.c_int (0)
    _checkRaise ("KN_get_mip_number_solves", _knitro.KN_get_mip_number_solves (kc, ctypes.byref (c_numSolves)))
    return c_numSolves.value

#---- KN_get_mip_abs_gap
_knitro.KN_get_mip_abs_gap.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_mip_abs_gap.restype = ctypes.c_int
def KN_get_mip_abs_gap (kc):
    c_absGap = ctypes.c_double (0)
    _checkRaise ("KN_get_mip_abs_gap", _knitro.KN_get_mip_abs_gap (kc, ctypes.byref (c_absGap)))
    return c_absGap.value

#---- KN_get_mip_rel_gap
_knitro.KN_get_mip_rel_gap.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_mip_rel_gap.restype = ctypes.c_int
def KN_get_mip_rel_gap (kc):
    c_relGap = ctypes.c_double (0)
    _checkRaise ("KN_get_mip_rel_gap", _knitro.KN_get_mip_rel_gap (kc, ctypes.byref (c_relGap)))
    return c_relGap.value

#---- KN_get_mip_incumbent_obj
_knitro.KN_get_mip_incumbent_obj.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_mip_incumbent_obj.restype = ctypes.c_int
def KN_get_mip_incumbent_obj (kc):
    c_incumbentObj = ctypes.c_double (0)
    _checkRaise ("KN_get_mip_incumbent_obj", _knitro.KN_get_mip_incumbent_obj (kc, ctypes.byref (c_incumbentObj)))
    return c_incumbentObj.value

#---- KN_get_mip_relaxation_bnd
_knitro.KN_get_mip_relaxation_bnd.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_mip_relaxation_bnd.restype = ctypes.c_int
def KN_get_mip_relaxation_bnd (kc):
    c_relaxBound = ctypes.c_double (0)
    _checkRaise ("KN_get_mip_relaxation_bnd", _knitro.KN_get_mip_relaxation_bnd (kc, ctypes.byref (c_relaxBound)))
    return c_relaxBound.value

#---- KN_get_mip_lastnode_obj
_knitro.KN_get_mip_lastnode_obj.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_mip_lastnode_obj.restype = ctypes.c_int
def KN_get_mip_lastnode_obj (kc):
    c_lastNodeObj = ctypes.c_double (0)
    _checkRaise ("KN_get_mip_lastnode_obj", _knitro.KN_get_mip_lastnode_obj (kc, ctypes.byref (c_lastNodeObj)))
    return c_lastNodeObj.value

#---- KN_get_mip_rel_gap
_knitro.KN_get_mip_incumbent_x.argtypes = [KN_context_ptr, ctypes.POINTER (ctypes.c_double)]
_knitro.KN_get_mip_incumbent_x.restype = ctypes.c_int
def KN_get_mip_incumbent_x (kc):
    nV = KN_get_number_vars (kc)
    c_x = (ctypes.c_double * nV) ()
    _checkRaise ("KN_get_mip_incumbent_x", _knitro.KN_get_mip_incumbent_x (kc, c_x))
    return _userArray (nV, c_x)
