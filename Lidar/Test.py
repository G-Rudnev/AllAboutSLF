import numpy as np
from numpy.random import rand as rand 
from numpy.random import randn as randn 
import time
import threading

import polylineCpp as pCpp

import math
from math import fabs as fabs
from math import pi as pi
_2pi = 2.0 * pi
from math import sqrt as sqrt
from math import sin as sin
from math import cos as cos
from math import tan as tan
from math import atan as atan
from math import atan2 as atan2

from Config import*
from RoboMath import*

import matplotlib.pyplot as plt

N = 3

polyline = Polyline(N)

id = pCpp.init(polyline, polyline.checkList, polyline.Nlines, (polyline.Nlines[0], ))

polyline[:2, 0] = [7.5486, 4.264]
polyline[:2, 1] = [7.5486, 26.1182]
polyline[:2, 2] = [27.3653, 12.7878]

ret_tup_0 = (0, )

p0 = np.array([8.2541, 4.0717])
p1 = np.array([7.5486, 26.1182])

pCpp.check_segment_intersections(id, p0, p1, 0, N - 1, True, 0, True, ret_tup_0)
pCpp.synchronize(id)

print(ret_tup_0)
print(polyline.checkList)


