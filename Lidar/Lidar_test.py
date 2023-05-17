import numpy as np
from numpy.random import rand as rand 
from numpy.random import randn as randn 
import time
import threading
from Lidar import Lidar

import math
from math import fabs as fabs
from math import pi as pi
_2pi = 2.0 * pi
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

    safety = float(globalPars["safety_scale"])
    half_width = float(globalPars["half_track"])
    half_length = float(globalPars["half_length"])

    @classmethod
    def plot(cls, ax):
        
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

        ax.plot(obb_pnts[0, :], obb_pnts[1, :], color = 'blue', linewidth = 1.0)

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

    def Calculate(self, eL2G : np.ndarray, prev_type, accuracy, latency, r00 = m.r0, iterateBy = None, eG2L : np.ndarray = None):
        """Calculates current frame to be executed, stores it in self.\n
        eL2G can be either 2D (3x3) or 3D (4x4) transformation.\n
        Returns false if and only if we iterate by iterateBy and reach the end or an error.\n
        Accuracy in parts (0..1) of m.r0, the smaller, the accurate.\n
        Latency is relative (0..1), the higher, the latencer to go on goal"""

        if (eG2L is None):
            eG2L = inv(eL2G)

        if (len(eL2G) == 4):
            rLC = np.array([np.dot(eG2L[0, :], self.L2G[:, 3]), np.dot(eG2L[1, :], self.L2G[:, 3])])
        else:
            rLC = np.array([np.dot(eG2L[0, :2], self.L2G[:2, 3]) + eG2L[0, 2], np.dot(eG2L[1, :2], self.L2G[:2, 3]) + eG2L[1, 2]])
        dist = norm(rLC)
        alphaDist = atan2(eL2G[0, 0] * self.L2G[1, 0] - eL2G[1, 0] * self.L2G[0, 0], eL2G[0, 0] * self.L2G[0, 0] + eL2G[1, 0] * self.L2G[1, 0])

        #if the goal reached by position and by angle
        if (iterateBy is not None and dist < 1.05 * r00 and fabs(alphaDist) <= m.alpha0):

            self.type = Route.AS_REQUIRED
            
            if (not iterateBy._Iterate(self)):     #the end of the route arc set or an error on route
                return False

            self.mode = self._mode

            if (fabs(self._alpha) > 0.0001):
                matmul(self.L2G, Transform_mat(self._s * sin(self._alpha) / self._alpha, self._s * (1.0 - cos(self._alpha)) / self._alpha, 0.0, 0.0, 0.0, self._alpha, 1), self.L2G)
            else:
                matmul(self.L2G, Transform_mat(self._s, 0.0, 0.0, 0.0, 0.0, self._alpha, 1), self.L2G)

            attractionVal = atan(rLC[1] / m.r0) / 2.0
            
            if (len(eL2G) == 4):
                rLC = np.array([np.dot(eG2L[0, :], self.L2G[:, 3]), np.dot(eG2L[1, :], self.L2G[:, 3])])
            else:
                rLC = np.array([np.dot(eG2L[0, :2], self.L2G[:2, 3]) + eG2L[0, 2], np.dot(eG2L[1, :2], self.L2G[:, 3]) + eG2L[1, 2]])
            dist = norm(rLC)
            alphaDist = atan2(eL2G[0, 0] * self.L2G[1, 0] - eL2G[1, 0] * self.L2G[0, 0], eL2G[0, 0] * self.L2G[0, 0] + eL2G[1, 0] * self.L2G[1, 0])

            if (fabs(alphaDist - _2pi - self._alpha) < fabs(alphaDist - self._alpha)):
                alphaDist -= _2pi
            elif (fabs(alphaDist + _2pi - self._alpha) < fabs(alphaDist - self._alpha)):
                alphaDist += _2pi

            if (self.direction):
            
                if (fabs(alphaDist) > 0.0001):
                    self.alpha = alphaDist + attractionVal * sin(alphaDist) / alphaDist
                else:
                    self.alpha = alphaDist + attractionVal

                self.l = dist

                if (fabs(self.alpha) > 0.0001):
                    self.s = self.l * fabs(self.alpha / 2.0 / sin(self.alpha / 2.0))
                else:
                    self.s = self.l

                if (self.s - self._s > m.r0):
                    self.s = self._s + m.r0
                    self.l = self.s * fabs(2.0 * sin(self.alpha / 2.0) / self.alpha)
                
            else:
                
                if (fabs(alphaDist) > 0.0001):
                    self.alpha = alphaDist - attractionVal * sin(alphaDist) / alphaDist
                else:
                    self.alpha = alphaDist - attractionVal

                self.l = -dist

                if (fabs(self.alpha) > 0.0001):
                    self.s = self.l * fabs(self.alpha / 2.0 / sin(self.alpha / 2.0))
                else:
                    self.s = self.l

                if (self._s - self.s > m.r0):
                    self.s = self._s - m.r0
                    self.l = self.s * fabs(2.0 * sin(self.alpha / 2.0) / self.alpha)

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
                        self.l = dist
                    else:
                        self.l = 0.0
                else:
                    if (not self.direction):
                        self.l = -dist
                    else:
                        self.l = 0.0

                if (r0 * r0 < ((rLC[0] - self.l * cos(self.alpha))**2 + (rLC[1] - self.l * sin(self.alpha))**2) ):
                    self.l /= 2.0

                if (fabs(self.alpha) > 0.0001):
                    self.s = self.l * self.alpha / sin(self.alpha)
                else:
                    self.s = self.l      

                self.alpha *= 2.0
            
            else:   #WE ARE FAR

                if (prev_type > Route.TO_GOAL_CLOSE):
                    self.type = prev_type   #to calculate related move
                else:
                    self.type = Route.TO_GOAL_FAR_ZERO

                self.mode = m.NORMAL

                if (self.direction):
                    self.l = r0
                else:
                    rLC[0] = -rLC[0]
                    rLC[1] = -rLC[1]
                    self.l = -r0 

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

                if (dist <= 2.0 * r00):
                    self.alpha = atan2(rLC[1] - (2.0 * (fabs(alphaFarEst) < pi / 2.0) - 1.0) * r0 * alphaDistAfter / pi * (1.0 + latency * (dist / r00 / 2.0) ** 2), rLC[0])
                else:
                    self.alpha = atan2(rLC[1] - (2.0 * (fabs(alphaFarEst) < pi / 2.0) - 1.0) * r0 * alphaDistAfter / pi * (1.0 + latency), rLC[0]) * ((2.0 * r00 / dist) ** latency)

                if (fabs(self.alpha) > 0.0001):
                    self.s = self.l * self.alpha / 2.0 / sin(self.alpha / 2.0)
                else:
                    self.s = self.l
        
        return True

    def plot(self, ax):
        ax.scatter(self.L2G[0, 3], self.L2G[1, 3], s = 50.0, c = 'green', alpha = 0.25, linewidths = 1.0)
        ax.plot([self.L2G[0, 3], self.L2G[0, 3] + 0.4 * self.L2G[0, 0]], [self.L2G[1, 3], self.L2G[1, 3] + 0.4 * self.L2G[1, 0]], color = 'green', linewidth = 1.0)

r = Route(0)
r.L2G = Transform_mat(3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
# r.L2G = Transform2D_mat(6.0 * rand(), 6.0 * rand(), randn() * pi / 2.0, 1)

rA = Route(0)

def foo(lid, ax, fig):

    time.sleep(0.2)

    spacingMin = e.safety * math.sqrt((e.half_length) ** 2 + (e.half_width) ** 2)
    spacingMean = 1.25 * spacingMin

    checkList = np.zeros([3, lid.N // 2])

    next = 1
    t0 = 0.0

    while plt.get_fignums():

        if (next):

            print(time.time() - t0)

            if (next == 2):
                print("Totally trapped - next rL2G")
                time.sleep(3.0)
            elif (next == 3):
                print("Totally trapped - no way out")
                time.sleep(3.0)
            elif (next == 4):
                print("The goal reached!")
                time.sleep(2.0)

            next = 0

            Nlines = 0

            while Nlines == 0:
                Nlines = lid.GetLinesXY()

            print(lid.linesXY.gapsIdxs[:lid.linesXY.Ngaps])

            e.L2G = np.eye(3)

            # xlim = ax.get_xlim()
            # ylim = ax.get_ylim()

            # ax.cla()

            # ax.set(xlim = xlim, ylim = ylim)
            # ax.set_aspect('equal')

            # lid.linesXY.plotLines(ax)
            # e.plot(ax)
            # r.plot(ax)
            # fig.canvas.draw_idle()

            trapped = False
            trappedNonGap = False
            goInGap = False
            gappyGoal = None
            gapsUsed = set()

            goInGapOnce = False
            trappedNonGapOnce = False

            t0 = time.time()

            xFar_i_reserve = Closest_pnt_on_lines(r.L2G[:2, 3], lid.linesXY, 0, Nlines)[0]

            cumAlpha = 0.0

        if (trapped):
            e.L2G = np.eye(3)

        basePnt = e.L2G[:2, 2]
        goalPnt = r.L2G[:2, 3]

        minDist = Closest_pnt_on_lines(basePnt, lid.linesXY, Nlines)[2]
        xNum = Check_segment_intersects_lines(basePnt, goalPnt, lid.linesXY, 0, Nlines, checkList)

        if (xNum != 0):

            iList_near = checkList[1, :xNum].argmin()
            xNear_i = int(checkList[0, iList_near])
            # xNear_t = checkList[1, iList_near]

            iList_far = checkList[1, :xNum].argmax()
            xFar_i = int(checkList[0, iList_far])
            xFar_t = checkList[1, iList_far]
            xFar_isGap = (checkList[2, iList_far] != 0.0)

            numOfxGaps = int(np.sum(checkList[2, :xNum]))
            if (numOfxGaps == xNum and not goInGapOnce):
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
                    gappyGoal = None
                    gapsUsed.clear()
                    trappedNonGap = False
            elif(not goInGapCheck):
                gappyGoal = None
                gapsUsed.clear()

            if (not goInGapCheck):
                goInGap = trappedNonGap
                goInGapOnce = True

        elif (trapped):
            trappedNonGap = True
            trappedNonGapOnce = True
        else:
            goInGap = goInGapCheck


        if (trappedNonGap):

            goInGap = True

            if (not trapped and gappyGoal is not None):
                gap_i = Closest_pnt_on_lines(gappyGoal, lid.linesXY, Nlines, True)[0]
            else:
                if (not lid.linesXY.closed and not gapsUsed.issuperset(lid.linesXY.gapsIdxs)):
                    gapsUsed_ = gapsUsed.copy()
                    gapsUsed.update({0, Nlines - 1})
                    gap_i = Closest_pnt_on_lines(basePnt, lid.linesXY, Nlines, True, gapsUsed_)[0]
                else:
                    gap_i = Closest_pnt_on_lines(basePnt, lid.linesXY, Nlines, True, gapsUsed)[0]

            if (gap_i < 0):
                next = 2
                continue

        elif (goInGap):

            if (not trapped and gappyGoal is not None):

                gap_i = Closest_pnt_on_lines(gappyGoal, lid.linesXY, Nlines, True)[0]
            
            else:

                noWayInc = False
                noWayDec = False

                while (True):

                    if (xFar_isGap and (xFar_i + 1 not in gapsUsed)):
                        gap_i = xFar_i + 1
                        break

                    xFar_pnt = basePnt + xFar_t * (goalPnt - basePnt)

                    if (lid.linesXY.closed and xFar_i == Nlines - 1):
                        k = 0
                    else:
                        k = xFar_i + 1

                    to_break = False

                    while (True):

                        if (k > Nlines - 1):
                            kInc = Nlines - 1
                            noWayInc = True
                            break
                        
                        k += 1

                        pnt, shift = lid.linesXY.GetPnt(k, -1)  #yep, -1

                        if (pnt is None):
                            kInc = Nlines - 1
                            pnt = lid.linesXY.GetPnt(Nlines - 1)[0]
                            to_break = True
                        elif (shift != 0):
                            kInc = k
                            to_break = True

                        if (to_break):
                            if (kInc in gapsUsed):
                                to_break = False
                                continue
                            distInc = norm(pnt - xFar_pnt)
                            break
                            
                    k = xFar_i
                    to_break = False

                    while (True):

                        if (k < 0):
                            kDec = 0
                            noWayDec = True
                            break

                        k -= 1

                        pnt, shift = lid.linesXY.GetPnt(k, 1)   #yep, 1

                        if (pnt is None):
                            kDec = 0
                            pnt = lid.linesXY.GetPnt(0)[0]
                            to_break = True
                        elif (shift != 0):
                            kDec = k
                            to_break = True

                        if (to_break):
                            if (kDec in gapsUsed):
                                to_break = False
                                continue
                            distDec = norm(pnt - xFar_pnt)
                            break
                    
                    if (noWayInc and noWayDec):
                        break
                    elif (noWayDec):
                        gap_i = kInc
                    elif (noWayInc):
                        gap_i = kDec
                    elif ((kDec == 0) == (kInc == Nlines - 1)):
                        if (distInc <= distDec):
                            gap_i = kInc
                        else:
                            gap_i = kDec
                    elif (kDec == 0):
                        gap_i = kInc
                    else:
                        gap_i = kDec

                    break

                if (noWayInc and noWayDec):
                    if (trappedNonGapOnce):
                        next = 2
                    else:
                        next = 3
                    continue

        freeWay = False
        invSide = False
        edge = False

        while (goInGap):

            to_break = False
            edge = (gap_i == 0 or gap_i == Nlines - 1)

            if (not edge and xFar_i == gap_i - 1):
                if (xNum > 1 and numOfxGaps != xNum):
                    goInGap = False
                    break
                else:
                    freeWay = True
                    to_break = True

            if (edge):
                gappyGoal = lid.linesXY.GetPnt(gap_i, homogen = True)[0]
            else:
                pnt_L = lid.linesXY.GetPnt(gap_i - 1, homogen = True)[0]
                pnt_R = lid.linesXY.GetPnt(gap_i + 1, homogen = True)[0]
                gappyGoal = (pnt_L + pnt_R) / 2.0

            gapsUsed.add(gap_i)

            if (to_break):
                break

            if (edge):

                xNear_i = xFar_i
                xFar_i = gap_i

            else:

                xNum_ = Check_segment_intersects_lines(basePnt, pnt_L, lid.linesXY, 0, gap_i - 2, ignoreGaps = True)
                xNum_ = Check_segment_intersects_lines(basePnt, pnt_L, lid.linesXY, gap_i + 1, Nlines - (gap_i == 1), num = xNum_, ignoreGaps = True)

                xNum_ = Check_segment_intersects_lines(basePnt, pnt_R, lid.linesXY, 0, gap_i - 1, num = xNum_, ignoreGaps = True)
                xNum_ = Check_segment_intersects_lines(basePnt, pnt_R, lid.linesXY, gap_i + 2, Nlines, num = xNum_, ignoreGaps = True)

                goalPnt = gappyGoal[:2]

                if (xNum_ != 0):

                    xNum = Check_segment_intersects_lines(basePnt, goalPnt, lid.linesXY, 0, gap_i - 1, checkList)
                    xNum = Check_segment_intersects_lines(basePnt, goalPnt, lid.linesXY, gap_i + 1, Nlines, checkList, xNum)

                    if (xNum != 0):

                        iList_near = checkList[1, :xNum].argmin()
                        xNear_i = int(checkList[0, iList_near])
                        
                        if (xNum % 2):
                            xFar_i = gap_i - 1
                        else:
                            iList_far = checkList[1, :xNum].argmax()
                            xFar_i = int(checkList[0, iList_far])

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
            
            alpha = 0.0
            
            fwdDist = 2 * e.half_length
            if (fwdDist > goalDist):
                fwdDist = goalDist
            
            frameVector = r.L2G[:2, 0]

        else:

            if (xFar_i > xNear_i):
                N = xFar_i - xNear_i
            else:
                N = Nlines - (xNear_i - xFar_i)
            
            fullCircle = 0

            #INCREMENT
            it = xNear_i + 1
            v = lid.linesXY.GetPnt(it, 1)[0] - basePnt

            alpha = atan2(goalVector[0] * v[1] - goalVector[1] * v[0], goalVector[0] * v[0] + goalVector[1] * v[1])
            maxAlphaInc = alpha
            sideInc = 2 * (alpha >= 0.0) - 1
            distInc = norm(v)
            itInc = it

            if (invSide):
                sideInc = -sideInc
                v_ = lid.linesXY.GetPnt(gap_i - 1, -1)[0] - basePnt
                limAlpha = fabs(atan2(goalVector[0] * v_[1] - goalVector[1] * v_[0], goalVector[0] * v_[0] + goalVector[1] * v_[1]))
            else:
                limAlpha = _2pi

            for _ in range(N - 1):

                it += 1
                pnt, shift = lid.linesXY.GetPnt(it, 1)

                if (pnt is None):
                    fullCircle = 1
                    break
                elif (shift != 0):
                    continue

                u = v
                v = pnt - basePnt

                alpha += atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
                shiftL = lid.linesXY.GetPnt(it - 1)[1]
                shiftR = lid.linesXY.GetPnt(it + 1)[1]
                if (shiftL == 0 or shiftR == 0):
                    if (sideInc * alpha >= limAlpha):
                        fullCircle = 1
                        break
                    elif (sideInc * alpha > sideInc * maxAlphaInc):
                        distInc += norm(pnt - lid.linesXY.GetPnt(itInc)[0])
                        maxAlphaInc = alpha
                        itInc = it

            distInc += norm(r.L2G[:2, 3] - lid.linesXY.GetPnt(itInc)[0]);            

            #DECREMENT
            N = Nlines - N + edge
            it = xNear_i - invSide
            v = lid.linesXY.GetPnt(it, -1)[0] - basePnt
            alpha = atan2(goalVector[0] * v[1] - goalVector[1] * v[0], goalVector[0] * v[0] + goalVector[1] * v[1])
            maxAlphaDec = alpha
            sideDec = -sideInc
            distDec = norm(v)
            itDec = it

            if (invSide):
                v_ = lid.linesXY.GetPnt(gap_i + 1, 1)[0] - basePnt
                limAlpha = fabs(atan2(goalVector[0] * v_[1] - goalVector[1] * v_[0], goalVector[0] * v_[0] + goalVector[1] * v_[1]))
            else:
                limAlpha = _2pi

            for _ in range(N - 1):

                it -= 1
                pnt, shift = lid.linesXY.GetPnt(it, -1)

                if (pnt is None):
                    fullCircle = -1
                    break
                elif (shift != 0):
                    continue

                u = v
                v = pnt - basePnt

                alpha += atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
                shiftL = lid.linesXY.GetPnt(it - 1)[1]
                shiftR = lid.linesXY.GetPnt(it + 1)[1]
                if (shiftL == 0 or shiftR == 0):
                    if (sideDec * alpha >= limAlpha):
                        fullCircle = -1
                        break
                    elif (sideDec * alpha > sideDec * maxAlphaDec):
                        distDec += norm(pnt - lid.linesXY.GetPnt(itDec)[0])
                        maxAlphaDec = alpha
                        itDec = it

            distDec += norm(r.L2G[:2, 3] - lid.linesXY.GetPnt(itDec, -1)[0])

            #VECTOR DEFINITION
            if (fullCircle < 0 or (fullCircle == 0 and distInc <= distDec)):
                alpha = maxAlphaInc
                side = sideInc
                it = itInc
            else:
                alpha = maxAlphaDec
                side = sideDec
                it = itDec

            v = lid.linesXY.GetPnt(it)[0] - basePnt
            if (side > 0):
                subGoalVector = v + np.array([-spacingMean * v[1], spacingMean * v[0]]) / norm(v)   # + normal to v
            else:
                subGoalVector = v + np.array([spacingMean * v[1], -spacingMean * v[0]]) / norm(v)

            subGoalDist = norm(subGoalVector)
            subGoalVector /= subGoalDist

            alpha += atan2(v[0] * subGoalVector[1] - v[1] * subGoalVector[0], v[0] * subGoalVector[0] + v[1] * subGoalVector[1])

            fwdDist = 2 * e.half_length
            # if (fwdDist > subGoalDist):
            #     fwdDist = subGoalDist

            frameVector = v


        if (minDist > spacingMean and pureFreeWay):
            latency = 1.0 - (subGoalDist / spacingMean / 10.0)
            if (latency < 0.2):
                latency = 0.2
        else:
            latency = 0.2

        dAlpha = None
        # dAlpha0 = (2 * invSide - 1) * alpha / 5.0
        dAlpha0 = alpha / -5.0
        if (dAlpha0 >= 0.0):
            if (dAlpha0 < 0.04):
                dAlpha0 = 0.04
        else:
            if (dAlpha0 > -0.04):
                dAlpha0 = 0.04
        

        alphaMax = -inf
        alphaMin = inf

        if (not freeWay):
            X = Check_segment_intersects_lines(basePnt, basePnt + subGoalDist * subGoalVector, lid.linesXY, 0, Nlines, ignoreGaps = True)
        else:
            X = 0

        while (True):

            if (dAlpha is not None):
                alpha += dAlpha

                trapped = True
                if (alpha > alphaMax):
                    alphaMax = alpha
                    trapped = False
                if (alpha < alphaMin):
                    alphaMin = alpha
                    trapped = False
                if (trapped or fabs(dAlpha) < 1e-4):
                    # print('Trapped !!!')
                    trapped = True
                    break

                subGoalVector = np.array([subGoalVector[0] * cos(dAlpha) - subGoalVector[1] * sin(dAlpha), \
                                        subGoalVector[0] * sin(dAlpha) + subGoalVector[1] * cos(dAlpha)])
                
            else:
                trapped = False

            framePnt = basePnt + fwdDist * subGoalVector
            # ax.scatter(framePnt[0], framePnt[1], marker = '*')
            # fig.canvas.draw_idle()

            if (not freeWay):
                if (Check_segment_intersects_lines(basePnt, framePnt, lid.linesXY, 0, Nlines, ignoreGaps = True)):
                    dAlpha = alpha / -5.0
                    continue

            _, frameClosestPnt, frameMinDist = Closest_pnt_on_lines(framePnt, lid.linesXY, Nlines)

            if (frameMinDist < spacingMin):
                if (X == 0):
                    baseVector = frameClosestPnt - basePnt
                    # kDst = 1.0 - minDist / spacingMin
                    # if (kDst < 0.25):
                    #     kDst = 0.25
                    # print(kDst)
                    dAlpha = 0.33 * atan2(baseVector[0] * subGoalVector[1] - baseVector[1] * subGoalVector[0], baseVector[0] * subGoalVector[0] + baseVector[1] * subGoalVector[1])
                else:
                    dAlpha = dAlpha0
                continue

            if (dAlpha is not None):
                frameVector = subGoalVector

            break

        if (trapped):
            if (not goInGap):
                if (goInGapOnce or trappedNonGapOnce):
                    next = 2
                    continue
            cumAlpha = 0.0
            continue

        rA.L2G[:2, 0] = frameVector * (2 * rA.direction - 1)
        rA.L2G[:2, 3] = framePnt

        rA.Calculate(e.L2G, m.type, 1.0, latency, e.half_length)

        if (minDist < 1.2 * spacingMin):
            if (fabs(rA.alpha) > pi / 1.5):
                rA.l = 0.0
            else:
                rA.l *= (1.0 - 1.5 * fabs(rA.alpha) / pi)
            
        cumAlpha += fabs(rA.alpha)
        if (cumAlpha > 3.0 * _2pi):
            if (edge):
                next = 2
                continue
            else:
                trapped = True
                if (not goInGap):
                    if (goInGapOnce or trappedNonGapOnce):
                        next = 2
                        continue
                cumAlpha = 0.0
                continue

        if (fabs(rA.alpha) > 0.0001):
            m.LT = Transform2D_mat(rA.l * cos(rA.alpha / 2.0), rA.l * sin(rA.alpha / 2.0), rA.alpha, 1)
        else:
            m.LT = Transform2D_mat(rA.l, 0.0, rA.alpha, 1)

        matmul(e.L2G, m.LT, e.L2G)

        if (norm(e.L2G[:2, 2] - r.L2G[:2, 3]) < spacingMean):
            next = 4

        # e.plot(ax)
        # fig.canvas.draw_idle()

        # time.sleep(0.01)

def main():

    fig = plt.figure()
    ax = plt.axes()
    ax.set(xlim = (-1, 5), ylim = (-5, 5))
    ax.set_aspect('equal')

    lid0 = Lidar.Create(0)

    while not lid0.Start():
        time.sleep(1.0)

    threading.Thread(target=foo, args = (lid0, ax, fig)).start()

    plt.show()

    exit()

if __name__ == "__main__":
    main()