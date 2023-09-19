import numpy as np
from numpy.random import rand as rand 
from numpy.random import randn as randn 
import time
import threading
from Lidar import Lidar

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

class e:

    L2G = None

    half_length = float(globalPars["half_length"])
    half_width = float(globalPars["half_width"])
    length = 2.0 * half_length
    width = 2.0 * half_width
    radius = sqrt(half_length ** 2 + half_width ** 2)
    safety = 1.0 + float(globalPars["safetyBox"]) / 100.0
    safety_half_length = half_length * safety
    safety_half_width = half_width * safety
    safety_length = length * safety
    safety_width = width * safety
    safety_radius = radius * safety

    @classmethod
    def plot(cls, ax, color_ = 'blue'):
        
        obb_pnts = np.array([
        [cls.L2G[0, 2] + cls.L2G[0, 0] * cls.half_length + cls.L2G[0, 1] * cls.half_width, \
            cls.L2G[0, 2] + cls.L2G[0, 0] * cls.half_length - cls.L2G[0, 1] * cls.half_width, \
                cls.L2G[0, 2] - cls.L2G[0, 0] * cls.half_length - cls.L2G[0, 1] * cls.half_width, \
                    cls.L2G[0, 2] - cls.L2G[0, 0] * cls.half_length + cls.L2G[0, 1] * cls.half_width, \
                        cls.L2G[0, 2] + cls.L2G[0, 0] * cls.half_length + cls.L2G[0, 1] * cls.half_width], \
            \
        [cls.L2G[1, 2] + cls.L2G[1, 0] * cls.half_length + cls.L2G[1, 1] * cls.half_width, \
            cls.L2G[1, 2] + cls.L2G[1, 0] * cls.half_length - cls.L2G[1, 1] * cls.half_width, \
                cls.L2G[1, 2] - cls.L2G[1, 0] * cls.half_length - cls.L2G[1, 1] * cls.half_width, \
                    cls.L2G[1, 2] - cls.L2G[1, 0] * cls.half_length + cls.L2G[1, 1] * cls.half_width, \
                        cls.L2G[1, 2] + cls.L2G[1, 0] * cls.half_length + cls.L2G[1, 1] * cls.half_width] \
        ])

        ax.plot(obb_pnts[0, :], obb_pnts[1, :], color = color_, linewidth = 1.0)

class m:

    ##Move modes
    NORMAL = 0
    STOP = 1
    """This is only about the end of the frame - will the robo stop in the end of it or not.\n
    An arc [1.0, 0.5, 0.0] will move the robo for 0.5 m and than stop him, no matter was the robo staying or moving before"""
    JUMP = 2
    """In loop mode the r.ARC_Set[1, it] is a next iterator"""
    HOLD = 3
    """No movements or stop as it is no movements expected, whatever, it's mean: mode = STOP, s = alpha = 0.0"""

    r0 = float(globalPars["r0"])
    alpha0 = pi / 8.0

    mode = STOP
    s, alpha = 0.0, 0.0
    """For backward moving: s - positive, alpha - global oriented.\n
    Actually both parameters consider globally as the length of an arc and the rotation angle, respectively.\n
    Direction defines by s sign."""
    l = 0.0
    """Chord for frame calculations."""
    type = 0
    """Move type by possible route activities"""

    LT = np.eye(4)

class Route:

    #Possible types of route activity
    AS_REQUIRED = 0
    """We have reached the goal, iterate next and go on as route requires"""
    TO_GOAL_CLOSE = 1
    """We are close to the goal"""
    TO_GOAL_FAR_ZERO = 2
    """We are far to the goal and we are going to reach it normally (straight)"""
    TO_GOAL_FAR_POSITIVE = 3
    """We are far to the goal and we are going to reach it from the right"""
    TO_GOAL_FAR_NEGATIVE = 4
    """We are far to the goal and we are going to reach it from the left"""

    def __init__(self, N):
        """
            Sets self.it and self.NSteps to 0, arg N is not a self.NSteps, it is max route length.
        """

        self.L2G0 = np.eye(4)
        """Initial route transformation"""
        self.L2G = np.eye(4)
        """Current route transformation"""

        self.ARC_Set = np.zeros([3, N])
        """[mode, distance (typically), angle (typically)]."""

        self._mode = m.STOP        
        """Current required operational mode"""
        self._s = 0.0          
        """Current required arc length"""
        self._alpha = 0.0     
        """Current required route angle"""

        self.mode = m.STOP        
        """Current operational mode to use"""
        self.s = 0.0          
        """Current arc length to use"""
        self.alpha = 0.0     
        """Current angle to use"""
        self.l = 0.0     
        """Current chord to use"""
        self.direction = True
        """Current direction to use, true - forward, false - backward"""
        self.type = Route.AS_REQUIRED
        """Considered move type"""

        self.NSteps = 0
        self.it = 0

        self.ready = False

    def Calculate(self, eL2G : np.ndarray, prev_type, accuracy, latencyDist, iterateBy = None, eG2L : np.ndarray = None, r00 = m.r0):
            """Calculates current frame to be executed, stores it in self.\n
            eL2G can be either 2D (3x3) or 3D (4x4) transformation.\n
            Returns false if and only if we iterate by iterateBy and reach the end or an error.\n
            accuracy in parts (0..1) of m.r0, the smaller, the accurate.\n
            latencyDist is the distance to goal of the medium smoothness and have to be > 0.0 (the farther the less smoothnes)"""

            if (eG2L is None):
                eG2L = inv(eL2G)

            if (len(eL2G) == 4):
                rLC = np.array([np.dot(eG2L[0, :], self.L2G[:, 3]), np.dot(eG2L[1, :], self.L2G[:, 3])])
            else:
                rLC = np.array([np.dot(eG2L[0, :2], self.L2G[:2, 3]) + eG2L[0, 2], np.dot(eG2L[1, :2], self.L2G[:2, 3]) + eG2L[1, 2]])
            dist = norm(rLC)
            alphaDist = atan2(eL2G[0, 0] * self.L2G[1, 0] - eL2G[1, 0] * self.L2G[0, 0], eL2G[0, 0] * self.L2G[0, 0] + eL2G[1, 0] * self.L2G[1, 0])

            #if the goal reached by position and by angle
            if (iterateBy is not None and dist < 1.05 * r00): # and fabs(alphaDist) <= m.alpha0):

                self.type = Route.AS_REQUIRED
                
                if (not iterateBy.Iterate(self)):     #the end of the route arc set or an error on route
                    return False

                self.mode = self._mode

                if (fabs(self._alpha) > 1e-4):
                    matmul(self.L2G, Transform_mat(self._s * sin(self._alpha) / self._alpha, self._s * (1.0 - cos(self._alpha)) / self._alpha, 0.0, 0.0, 0.0, self._alpha, 1), self.L2G)
                else:
                    matmul(self.L2G, Transform_mat(self._s, 0.0, 0.0, 0.0, 0.0, self._alpha, 1), self.L2G)

                attractionVal = atan(rLC[1] / r00) / 2.0
                
                if (len(eL2G) == 4):
                    rLC = np.array([np.dot(eG2L[0, :], self.L2G[:, 3]), np.dot(eG2L[1, :], self.L2G[:, 3])])
                else:
                    rLC = np.array([np.dot(eG2L[0, :2], self.L2G[:2, 3]) + eG2L[0, 2], np.dot(eG2L[1, :2], self.L2G[:2, 3]) + eG2L[1, 2]])
                dist = norm(rLC)
                alphaDist = atan2(eL2G[0, 0] * self.L2G[1, 0] - eL2G[1, 0] * self.L2G[0, 0], eL2G[0, 0] * self.L2G[0, 0] + eL2G[1, 0] * self.L2G[1, 0])

                if (fabs(alphaDist - _2pi - self._alpha) < fabs(alphaDist - self._alpha)):
                    alphaDist -= _2pi
                elif (fabs(alphaDist + _2pi - self._alpha) < fabs(alphaDist - self._alpha)):
                    alphaDist += _2pi

                if (self.direction):
                
                    if (fabs(alphaDist) > 1e-4):
                        self.alpha = alphaDist + attractionVal * sin(alphaDist) / alphaDist
                    else:
                        self.alpha = alphaDist + attractionVal

                    if (fabs(self.alpha) > 1e-4):
                        self.s = dist * fabs(self.alpha / 2.0 / sin(self.alpha / 2.0))
                    else:
                        self.s = dist

                    if (self.s - self._s > r00):
                        self.s = self._s + r00
                    
                else:
                    
                    if (fabs(alphaDist) > 1e-4):
                        self.alpha = alphaDist - attractionVal * sin(alphaDist) / alphaDist
                    else:
                        self.alpha = alphaDist - attractionVal

                    if (fabs(self.alpha) > 1e-4):
                        self.s = -dist * fabs(self.alpha / 2.0 / sin(self.alpha / 2.0))
                    else:
                        self.s = -dist

                    if (self._s - self.s > r00):
                        self.s = self._s - r00

            #if the goal is not reached we have to get to it
            else:

                if (self.direction):
                    r0 = (r00 + rLC[0]) * accuracy
                else:
                    r0 = (r00 - rLC[0]) * accuracy
                
                if (r0 > r00):
                    r0 = r00
                elif (r0 <= r00 * accuracy):
                    r0 = r00 * accuracy
                
                if (dist < r0):  #WE ARE CLOSE
                                    
                    self.type = Route.TO_GOAL_CLOSE

                    self.mode = self._mode
                    self.alpha = alphaDist / 2.0

                    if ((rLC[0] - dist * cos(self.alpha))**2 + (rLC[1] - dist * sin(self.alpha))**2 <= \
                        (rLC[0] + dist * cos(self.alpha))**2 + (rLC[1] + dist * sin(self.alpha))**2):
                        if (self.direction):    #move direction is only as the route defines
                            l = dist
                        else:
                            l = 0.0
                    else:
                        if (not self.direction):
                            l = -dist
                        else:
                            l = 0.0

                    if (r0 * r0 < ((rLC[0] - l * cos(self.alpha))**2 + (rLC[1] - l * sin(self.alpha))**2) ):
                        l /= 2.0

                    if (fabs(self.alpha) > 1e-4):
                        self.s = l * self.alpha / sin(self.alpha)
                    else:
                        self.s = l      

                    self.alpha *= 2.0
                
                else:   #WE ARE FAR

                    if (prev_type > Route.TO_GOAL_CLOSE):
                        self.type = prev_type   #to calculate related move
                    else:
                        self.type = Route.TO_GOAL_FAR_ZERO

                    self.mode = m.NORMAL

                    if (self.direction):
                        l = r0
                    else:
                        rLC[0] = -rLC[0]
                        rLC[1] = -rLC[1]
                        l = -r0

                    alphaFarEst = atan2(rLC[1], rLC[0])
                    alphaDistAfter = alphaDist - alphaFarEst

                    if (self.type == Route.TO_GOAL_FAR_ZERO):
                        if (alphaDistAfter > 0.9 * pi):
                            self.type = Route.TO_GOAL_FAR_POSITIVE
                        elif (alphaDistAfter < -0.9 * pi):
                            self.type = Route.TO_GOAL_FAR_NEGATIVE
                    elif (fabs(alphaDistAfter) < 0.75 * pi):
                        self.type = Route.TO_GOAL_FAR_ZERO
                    
                    if (self.type == Route.TO_GOAL_FAR_NEGATIVE and alphaDistAfter > 0.0):
                        alphaDistAfter -= _2pi
                    elif (self.type == Route.TO_GOAL_FAR_POSITIVE and alphaDistAfter < 0.0):
                        alphaDistAfter += _2pi

                    if (alphaDistAfter > pi):
                        alphaDistAfter -= _2pi
                    elif (alphaDistAfter < -pi):
                        alphaDistAfter += _2pi

                    latency = latencyDist / (dist + latencyDist)

                    if (dist <= 2.0 * r00):
                        self.alpha = atan2(rLC[1] - (2.0 * (fabs(alphaFarEst) < pi / 2.0) - 1.0) * r0 * alphaDistAfter / pi * latency * (1.0 + (dist / r00 / 2.0) ** 2), rLC[0])
                    else:
                        self.alpha = atan2(rLC[1] - (2.0 * (fabs(alphaFarEst) < pi / 2.0) - 1.0) * r0 * alphaDistAfter / pi * latency * 2.0, rLC[0]) * ((2.0 * r00 / dist) ** latency)

                    if (fabs(self.alpha) > 1e-4):
                        self.s = l * self.alpha / 2.0 / sin(self.alpha / 2.0)
                    else:
                        self.s = l

                # if (self.alpha > 0.0):
                #     if (self.alpha > pi / 3.0):
                #         self.s *= pi / 3.0 / self.alpha
                #         self.alpha = pi / 3.0
                # else:
                #     if (self.alpha < pi / -3.0):
                #         self.s *= pi / -3.0 / self.alpha
                #         self.alpha = pi / -3.0
                    
            return True

    def plot(self, ax):
        ax.scatter(self.L2G[0, 3], self.L2G[1, 3], s = 50.0, c = 'green', alpha = 0.25, linewidths = 1.0)
        ax.plot([self.L2G[0, 3], self.L2G[0, 3] + 0.4 * self.L2G[0, 0]], [self.L2G[1, 3], self.L2G[1, 3] + 0.4 * self.L2G[1, 0]], color = 'green', linewidth = 1.0)

e.L2G = np.eye(3)

r = Route(0)
r.L2G = Transform_mat(-1.0, -0.02, 0.0, 0.0, 0.0, pi, 1)
# r.L2G = Transform2D_mat(6.0 * rand(), 6.0 * rand(), randn() * pi / 2.0, 1)

rA = Route(0)

def foo(lid, ax, fig):

    time.sleep(0.2)

    polyline = Polyline(lid.N)

    id = pCpp.init(polyline, polyline.checkList, polyline.Nlines, (polyline.Nlines[0], ))

    pnt = np.array([1.0, 1.0])

    ret_tup_0 = (0, )
    ret_tup_3 = (0, np.zeros([2]), 0.0)

    while plt.get_fignums():

        Nlines = 0

        while Nlines == 0:
            Nlines = lid.GetLinesXY(polyline)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        ax.cla()
        
        if (norm(r.L2G[:2, 3] - e.L2G[:2, 2]) < m.r0):
            e.L2G = np.eye(3)

        ax.set(xlim = xlim, ylim = ylim)
        ax.set_aspect('equal')

        polyline.plotLines(ax)
        r.plot(ax)

        seg_from = 1
        seg_to = Nlines - 1

        pCpp.check_if_obb_intersection(id, e.L2G, e.half_length, e.half_width, 1, Nlines - 1, ret_tup_0)
        pCpp.synchronize(id)

        num = ret_tup_0[0]
        print(num)
        
        if (num > 0):
            color = 'red'
        else:
            color = 'blue'

        e.plot(ax, color)

        fig.canvas.draw_idle()
        
        r.Calculate(e.L2G, r.type, 0.5, 0.1)
        if (fabs(r.alpha) > 1e-4):
            np.matmul(e.L2G, Transform2D_mat(r.s * sin(r.alpha) / r.alpha, r.s * (1.0 - cos(r.alpha)) / r.alpha, r.alpha, 1), e.L2G)
        else:
            np.matmul(e.L2G, Transform2D_mat(r.s, 0.0, r.alpha, 1), e.L2G)
        
        continue

        GO_ON = 0
        START = 1
        REACHED = 2
        TRAPPED_FREE = 3
        NOGAPS_NONGAP = 4
        NOGAPS_GAP = 5

        spacingMin = math.sqrt((e.half_length) ** 2 + (e.half_width) ** 2)
        spacingMean = e.safety * spacingMin

        edgeGaps = (polyline.gapsIdxs[0], polyline.gapsIdxs[polyline.Ngaps - 1])

        seg_from = polyline.edgeShift
        seg_to = Nlines - seg_from - 1

        gapsUsed = set()

        trapped = False
        trappedNonGap = False
        goInGap = False

        gappyGoal = None
        cumAlpha = 0.0

        e.L2G = np.eye(3)
        # r.L2G = ...

        xFar_i_reserve = polyline.Closest_pnt(r.L2G[:2, 3], 0, Nlines)[0]

        next = START
        noGapsCounter = 0

        while True:

            if (next):

                if (next == REACHED):
                    print("The goal is reached!")
                    break
                elif (next == TRAPPED_FREE):
                    print("Trapped in freeWay")
                elif (next > TRAPPED_FREE):
                    if (next == NOGAPS_NONGAP):
                        print("No gaps in trappedNonGap")
                    elif (next == NOGAPS_GAP):
                        print("No gaps in goInGap")
                    if (noGapsCounter > 4):
                        break
                    elif (noGapsCounter > 2):
                        print('Multi no gaps')
                        gapsUsed = set()

                        trapped = False
                        trappedNonGap = False
                        goInGap = False

                        gappyGoal = None
                        cumAlpha = 0.0
                    noGapsCounter += 1

                gappyGoal = None
                cumAlpha = 0.0

                next = GO_ON

            basePnt = e.L2G[:2, 2]
            goalPnt = r.L2G[:2, 3]

            minDist = polyline.Closest_pnt(basePnt, seg_from, seg_to)[2]
            xNum = polyline.Check_segment_intersections(basePnt, goalPnt, 0, Nlines)

            if (xNum != 0):

                iList_near = polyline.checkList[1, :xNum].argmin()
                xNear_i = int(polyline.checkList[0, iList_near])
                # xNear_t = polyline.checkList[1, iList_near]

                iList_far = polyline.checkList[1, :xNum].argmax()
                xFar_i = int(polyline.checkList[0, iList_far])
                xFar_t = polyline.checkList[1, iList_far]
                xFar_isGap = (polyline.checkList[2, iList_far] != 0.0)

                numOfxGaps = int(np.sum(polyline.checkList[2, :xNum]))
                if (numOfxGaps == xNum):
                    goInGapCheck = True
                else:
                    goInGapCheck = ((xNum % 2) == 1)

                pureFreeWay = False

            else:

                xFar_i = xFar_i_reserve
                xFar_t = 1.0
                xFar_isGap = False

                numOfxGaps = 0
                goInGapCheck = False
                pureFreeWay = True


            if (goInGap):

                if (trappedNonGap):
                    if (goInGapCheck):
                        # gappyGoal = None
                        # gapsUsed.clear()
                        trappedNonGap = False
                elif(not goInGapCheck):
                    gappyGoal = None

                if (not goInGapCheck):
                    goInGap = trappedNonGap

            elif (trapped):
                trappedNonGap = True
            else:
                goInGap = goInGapCheck


            if (trappedNonGap):

                goInGap = True

                if (not trapped and gappyGoal is not None):
                    gap_i = polyline.Closest_pnt(gappyGoal, 0, Nlines, True)[0]
                else:
                    if (not gapsUsed.issuperset(polyline.gapsIdxs[1:polyline.Ngaps])):
                        gapsUsed_ = gapsUsed.copy()
                        gapsUsed_.update(edgeGaps)
                        gap_i = polyline.Closest_pnt(basePnt, 0, Nlines, True, gapsUsed_)[0]
                    else:
                        gap_i = polyline.Closest_pnt(basePnt, 0, Nlines, True, gapsUsed)[0]

                if (gap_i < 0):
                    next = NOGAPS_NONGAP    #нет доступных разрывов в активности trappedNonGap
                    continue

            elif (goInGap):

                if (not trapped and gappyGoal is not None):

                    gap_i = polyline.Closest_pnt(gappyGoal, 0, Nlines, True)[0]
                
                else:

                    noGapInc = False
                    noGapDec = False

                    while (True):

                        if (xFar_isGap and (xFar_i + 1 not in gapsUsed)):
                            gap_i = xFar_i + 1
                            break

                        if (xFar_i == Nlines - 1):
                            k = 0
                        else:
                            k = xFar_i + 1

                        to_break = False

                        while (True):

                            if (k >= Nlines):
                                kInc = Nlines - 1
                                noGapInc = True
                                break
                            
                            k += 1

                            pntInc, shift = polyline.GetPnt(k, -1)  #yep, -1

                            if (shift != 0):
                                kInc = k
                                to_break = True

                            if (to_break):
                                if (kInc in gapsUsed):
                                    to_break = False
                                    continue
                                break
                                
                        k = xFar_i
                        to_break = False

                        while (True):

                            if (k < 0):
                                kDec = 0
                                noGapDec = True
                                break

                            k -= 1

                            pntDec, shift = polyline.GetPnt(k, 1)   #yep, 1

                            if (shift != 0):
                                kDec = k
                                to_break = True

                            if (to_break):
                                if (kDec in gapsUsed):
                                    to_break = False
                                    continue
                                break

                        if (noGapInc and noGapDec):
                            break
                        elif noGapDec:
                            gap_i = kInc
                        elif noGapInc:
                            gap_i = kDec
                        elif ((kDec <= edgeGaps[0]) == (kInc >= edgeGaps[1])):
                            xFar_pnt = basePnt + xFar_t * (goalPnt - basePnt)
                            if (norm(pntInc - xFar_pnt) <= norm(pntDec - xFar_pnt)):
                                gap_i = kInc
                            else:
                                gap_i = kDec
                        elif (kDec <= edgeGaps[0]):
                            gap_i = kInc
                        else:
                            gap_i = kDec

                        break

                    if (noGapInc and noGapDec):
                        next = NOGAPS_GAP   #нет доступных разрывов в активности goInGap
                        continue

            freeWay = False
            invSide = False

            pnt_L = None
            pnt_R = None

            while (goInGap):

                to_break = False

                if (xFar_i == gap_i - 1):
                    to_break = True
                    if (xNum <= 1 or numOfxGaps == xNum):
                        freeWay = True

                pnt_L = polyline.GetPnt(gap_i - 1, homogen = True)[0]
                pnt_R = polyline.GetPnt(gap_i + 1, homogen = True)[0]

                gapsUsed.add(gap_i)

                if (to_break):
                    break

                xNum_ = polyline.Check_segment_intersections(basePnt, pnt_L, 0, gap_i - 2, ignoreGaps = True)
                xNum_ = polyline.Check_segment_intersections(basePnt, pnt_L, gap_i + 1, Nlines - (gap_i == 1), num = xNum_, ignoreGaps = True)

                xNum_ = polyline.Check_segment_intersections(basePnt, pnt_R, 0, gap_i - 1, num = xNum_, ignoreGaps = True)
                xNum_ = polyline.Check_segment_intersections(basePnt, pnt_R, gap_i + 2, Nlines, num = xNum_, ignoreGaps = True)

                goalPnt = (pnt_L[:2] + pnt_R[:2]) / 2.0

                if (xNum_ != 0):

                    xNum = polyline.Check_segment_intersections(basePnt, goalPnt, 0, gap_i - 1)
                    xNum = polyline.Check_segment_intersections(basePnt, goalPnt, gap_i + 1, Nlines, xNum)

                    if (xNum != 0):

                        iList_near = polyline.checkList[1, :xNum].argmin()
                        xNear_i = int(polyline.checkList[0, iList_near])
                        
                        if (xNum % 2):
                            xFar_i = gap_i - 1
                        else:
                            iList_far = polyline.checkList[1, :xNum].argmax()
                            xFar_i = int(polyline.checkList[0, iList_far])

                else:

                    xNear_i = gap_i
                    invSide = True

                break


            freeWay = (xNum == 0 or freeWay)

            goalVector = goalPnt - basePnt
            goalDist = norm(goalVector)
            goalVector /= goalDist
            
            if (freeWay):

                subGoalVector = goalVector
                subGoalDist = goalDist
                
                malpha = 0.0
                
                fwdDist = 2.0 * e.half_length
                if (fwdDist > goalDist):
                    fwdDist = goalDist
                
                if (goInGap):
                    frameVector = subGoalVector.copy() #можно не копировать
                else:
                    frameVector = r.L2G[:2, 0]

            else:

                if (xFar_i > xNear_i):
                    NInc = xFar_i - xNear_i
                    NDec = Nlines - NInc
                else:
                    NDec = xNear_i - xFar_i
                    NInc = Nlines - NDec

                if (NInc <= NDec):
                    inc = 1
                    N = NInc
                else:
                    inc = -1
                    N = NDec

                if (inc > 0):
                    idxCur = xNear_i + 1
                else:
                    idxCur = xNear_i - invSide

                v = polyline.GetPnt(idxCur, inc)[0] - basePnt

                malpha = alphaCur = atan2(goalVector[0] * v[1] - goalVector[1] * v[0], goalVector[0] * v[0] + goalVector[1] * v[1])
                if (alphaCur >= 0.0):
                    side = 1
                else:
                    side = -1
                idx = idxCur

                if (invSide):
                    side = -side
                    v_ = polyline.GetPnt(gap_i - inc, -inc)[0] - basePnt
                    limAlpha = fabs(atan2(goalVector[0] * v_[1] - goalVector[1] * v_[0], goalVector[0] * v_[0] + goalVector[1] * v_[1]))
                else:
                    limAlpha = _2pi

                for _ in range(N - 1):

                    idxCur += inc
                    pnt, shift = polyline.GetPnt(idxCur, inc)

                    if (shift != 0):
                        continue

                    u = v
                    v = pnt - basePnt

                    alphaCur += atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
                    shiftL = polyline.GetPnt(idxCur - 1)[1]
                    shiftR = polyline.GetPnt(idxCur + 1)[1]
                    if (shiftL == 0 or shiftR == 0):
                        if (side * alphaCur >= limAlpha):
                            break
                        elif (side * alphaCur > side * malpha):
                            malpha = alphaCur
                            idx = idxCur


                v = polyline.GetPnt(idx)[0] - basePnt
                if (side > 0):
                    subGoalVector = v + np.array([-spacingMean * v[1], spacingMean * v[0]]) / norm(v)   # + normal to v
                else:
                    subGoalVector = v + np.array([spacingMean * v[1], -spacingMean * v[0]]) / norm(v)

                subGoalDist = norm(subGoalVector)
                subGoalVector /= subGoalDist

                malpha += atan2(v[0] * subGoalVector[1] - v[1] * subGoalVector[0], v[0] * subGoalVector[0] + v[1] * subGoalVector[1])

                fwdDist = 2.0 * e.half_length
                if (fwdDist > subGoalDist):
                    fwdDist = subGoalDist

                frameVector = v.copy()


            dAlpha = None

            if (freeWay):
                xNum = 0
            else:
                xNum = polyline.Check_segment_intersections(basePnt, basePnt + subGoalDist * subGoalVector, 0, Nlines)

            if (xNum != 0):
                xNear_t = polyline.checkList[1, :xNum].min()
                xNearDist = xNear_t * subGoalDist
            else:
                xNearDist = inf

            alphaMax = -inf
            alphaMin = inf

            fwdDist = minDist
            if (fwdDist < e.length):
                fwdDist = e.length

            while (True):

                if (dAlpha is not None):

                    if (dAlpha > 0.0):
                        if (dAlpha < 0.1):
                            dAlpha = 0.1
                    elif (dAlpha > -0.1):
                        dAlpha = -0.1

                    malpha += dAlpha

                    trapped = True
                    if (malpha > alphaMax):
                        alphaMax = malpha
                        trapped = False
                    if (malpha < alphaMin):
                        alphaMin = malpha
                        trapped = False

                    if (trapped):
                        print("Trapped !!!")
                        break

                    subGoalVector = np.array([subGoalVector[0] * cos(dAlpha) - subGoalVector[1] * sin(dAlpha), \
                                            subGoalVector[0] * sin(dAlpha) + subGoalVector[1] * cos(dAlpha)])
                    
                else:
                    trapped = False

                framePnt = basePnt + fwdDist * subGoalVector
                # ax.scatter(framePnt[0], framePnt[1], marker = '*')
                # fig.canvas.draw_idle()

                if (not freeWay):
                    if (polyline.Check_segment_intersections(basePnt, framePnt, 0, Nlines, ignoreGaps = True)):
                        dAlpha = malpha / -4.0
                        continue

                _, frameClosestPnt, frameMinDist = polyline.Closest(framePnt, 0, Nlines)

                if (frameMinDist < spacingMin):
                    if (xNearDist > spacingMean):
                        baseVector = frameClosestPnt - basePnt
                        # kDst = 1.0 - minDist / spacingMin
                        # if (kDst < 0.25):
                        #     kDst = 0.25
                        # print(kDst)
                        if (invSide):
                            dAlpha = 0.33 * atan2(baseVector[0] * goalVector[1] - baseVector[1] * goalVector[0], baseVector[0] * goalVector[0] + baseVector[1] * goalVector[1])
                        else:
                            dAlpha = 0.33 * atan2(baseVector[0] * subGoalVector[1] - baseVector[1] * subGoalVector[0], baseVector[0] * subGoalVector[0] + baseVector[1] * subGoalVector[1])
                    else:
                        dAlpha = malpha / -4.0
                    continue

                if (dAlpha is not None):
                    frameVector = subGoalVector

                break

            if (trapped):
                if (not goInGap):
                    next = TRAPPED_FREE    #заблокирован на свободной видимости
                continue

            rA.L2G[:2, 0] = frameVector * (2 * rA.direction - 1)
            rA.L2G[:2, 3] = framePnt

            if (minDist > 2.0 * spacingMin and pureFreeWay):
                latency = 1.0 - (subGoalDist / spacingMean / 10.0)
                if (latency < 0.2):
                    latency = 0.2
            else:
                latency = 0.1

            rA.Calculate(e.L2G, m.type, 0.75, latency, e.half_length)

            if (minDist < 1.25 * spacingMin):
                if (fabs(rA.alpha) > pi / 1.5):
                    rA.l = 0.0
                else:
                    rA.l *= (1.0 - 1.5 * fabs(rA.alpha) / pi)
                
            cumAlpha += fabs(rA.alpha)
            if (cumAlpha > 3.0 * _2pi):
                trapped = True  #просто ищем следующий разрыв
                next = TRAPPED_FREE
                continue

            if (goInGap):
                gappyGoal = (pnt_L + pnt_R) / 2.0

            m.type = rA.type

            if (fabs(rA.alpha) > 0.0001):
                m.LT = Transform2D_mat(rA.l * cos(rA.alpha / 2.0), rA.l * sin(rA.alpha / 2.0), rA.alpha, 1)
            else:
                m.LT = Transform2D_mat(rA.l, 0.0, rA.alpha, 1)

            matmul(e.L2G, m.LT, e.L2G)

            if (norm(e.L2G[:2, 2] - r.L2G[:2, 3]) < e.safety_length):
                next = REACHED    #цель достигнута

            e.plot(ax)
            fig.canvas.draw_idle()

            time.sleep(0.1)

def main():

    fig = plt.figure()
    ax = plt.axes()
    ax.set(xlim = (-2, 2), ylim = (-2, 2))
    ax.set_aspect('equal')

    lid0 = Lidar.Create(0)

    while not lid0.Start():
        time.sleep(1.0)

    threading.Thread(target=foo, args = (lid0, ax, fig)).start()

    plt.show()

    exit()

if __name__ == "__main__":
    main()