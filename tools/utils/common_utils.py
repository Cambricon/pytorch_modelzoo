from __future__ import print_function
import sys
import os
import platform
import re
import gc
import types
import inspect
import argparse
import unittest
import warnings
import random
import contextlib
import socket
import time
import json
import numpy as np
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps
from itertools import product
from copy import deepcopy
from numbers import Number

class Timer:
    def __init__(self):
        self.start_ = datetime.now()

    def elapsed(self):
        duration_ = datetime.now() - self.start_
        return duration_.total_seconds()

class DumpData:
    def __init__(self, acc_standard='meanAp'):
        self.TIME = -1
        self.hardwareFps = -1
        self.endToEndFps = -1
        self.latencytime = -1
        self.acc_standard = acc_standard

    def dumpJson(self, imageNum, batch_size, top1, top5, meanAp, hardwaretime, endToEndTime):
        if hardwaretime != self.TIME:
            self.hardwareFps = imageNum / hardwaretime
            self.latencytime = hardwaretime / (imageNum / batch_size) * 1000
        if endToEndTime != self.TIME:
            self.endToEndFps = imageNum / endToEndTime
        if top1 != self.TIME:
            top1 = top1 / imageNum * 100
        if top5 != self.TIME:
            top5 = top5 / imageNum * 100

        print('latency: ' + str(self.latencytime))
        print('throughput: ' + str(self.endToEndFps))
        print(self.acc_standard + ':' + str(meanAp))

        if not os.getenv('OUTPUT_JSON_FILE'):
            return

        result={
                "Output":{
                    "Accuracy":{
                        "top1":'%.2f'%top1,
                        "top5":'%.2f'%top5,
                        self.acc_standard:'%.4f'%meanAp
                        },
                    "HostLatency(ms)":{
                        "average":'%.2f'%self.latencytime,
                        "throughput(fps)":'%.2f'%self.endToEndFps,
                        }
                    }
                }

        with open(os.getenv("OUTPUT_JSON_FILE"),"a") as outputfile:
            json.dump(result,outputfile,indent=4,sort_keys=True)
            outputfile.write('\n')

def get_precision_mode(data_dtype, quant_dtype):
    """
    get unified precision format
    Args:
        data_dtype: input data precision type, optional is [float32, float16]
        quant_dtype: quant precision type, optional is [int8, int16, no_quant]
    Returns:
        return precision mode
    """
    if data_dtype == "float32":
        if quant_dtype == "no_quant":
            return "force_float32"
        elif quant_dtype == "int8":
            return "qint8_mixed_float32"
        elif quant_dtype == "int16":
            return "qint16_mixed_float32"
        else:
            assert False, "unknown quant type {0}".format(quant_dtype)
    elif data_dtype == "float16":
        if quant_dtype == "no_quant":
            return "force_float16"
        elif quant_dtype == "int8":
            return "qint8_mixed_float16"
        elif quant_dtype == "int16":
            return "qint16_mixed_float16"
        else:
            assert False, "unknown quant type {0}".format(quant_dtype)
    else:
        assert False, "unknown data type {0}".format(data_dtype)

