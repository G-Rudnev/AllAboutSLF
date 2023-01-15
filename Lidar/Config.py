'''Override print(...), defines globalPars and mainDevicesPars'''

__all__ = ['globalPars', 'mainDevicesPars', 'print', 'RoboException', 'OperateCounter']

def getTokensFromFile(filename, dict):
    with open(filename, 'rt') as pars_ini:
        for feed in pars_ini.read(-1).split(';;')[:-1]:  #all useful data stored in the first line
            tokens = feed.split(":=")
            if len(tokens)==1:
                continue
            try:
                key = tokens[-2].split('\n')[-1]
                dict[key] = tokens[-1]
            except:
                continue

globalPars = {}
getTokensFromFile('Global.ini', globalPars)
mainDevicesPars = {}
getTokensFromFile('MainDevices.ini', mainDevicesPars)

class RoboException(Exception):
    pass

def _print_wrp(fc):
    def wrp(*args, **kwargs):
        isprint = True
        if kwargs.pop('log', False):
            if (args[0] == globalPars['MESSAGE']):
                isprint = False
            else:
                globalPars['MESSAGE'] = args[0]
        if kwargs.pop('exc', False):
            if isprint:
                fc(*args, **kwargs)
            raise RoboException(args[0])
        if isprint:
            fc(*args, **kwargs)
    return wrp

print = _print_wrp(print)
"""
Overrided! Use kwarg "log = True" for update globalPars['MESSAGE'] with args[0] of call. Use kwarg "exc = True" for raise exception of type Exception with args[0] of call message
"""

from threading import Lock
from time import sleep
class OperateCounter():
    """
    Thread-safe counter. 
    Get() returns True if inner counter > 0, otherwise returns False.
    """
    def __init__(self, init_value = 0):
        self._counter = init_value
        self._mutex = Lock()

    def Inc(self):
        with self._mutex:
            self._counter += 1

    def Dec(self):
        with self._mutex:
            self._counter -= 1

    def Set(self, default_value = 1):
        with self._mutex:
            self._counter = default_value

    def Get(self):
        with self._mutex:
            return (self._counter > 0)

    def GetValue(self):
        with self._mutex:
            return self._counter

    def Wait(self):
        while (self._counter > 0):
            sleep(0.00001)