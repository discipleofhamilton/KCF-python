# -*- coding: utf-8 -*-
import os, sys
import traceback

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