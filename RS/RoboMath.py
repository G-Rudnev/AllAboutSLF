import time
import math
from math import inf as inf
from math import pi as pi
from math import pow as pow
from math import tan as tan
from math import sin as sin
from math import cos as cos
from math import atan as atan
from math import atan2 as atan2

import numpy as np
from numpy import matmul as matmul
from numpy.linalg import norm as norm
from numpy.linalg import inv as inv

class Polyline(np.ndarray):

    """
        Arbitrary gapped and closed polyline.\n
        The first and the last pnts should be non-zero both.\n
        Two gaps should be separated by at least one non-gap pnt
    """

    def __new__(cls, N, *args, **kwargs):
        obj = np.zeros([3, N])
        obj[2, :] = 1.0
        return obj.view(cls)

    def __init__(self, N : int):
        """edgeShift is as it supposed by vectorization properties"""
        self.Nlines = np.array([N], dtype = np.int64)
        self.gapsIdxs = np.zeros([N], dtype = np.int64)
        self.Ngaps = 0

        self.checkList = np.zeros([3, N], dtype = np.float64)

    def Fill(self, polyline : np.ndarray, gapsIdxs : np.ndarray, Nlines : np.int64, Ngaps : int):
        self[:2, :Nlines] = polyline[:2, :Nlines]
        self.gapsIdxs[:Ngaps] = gapsIdxs[:Ngaps]
        self.Nlines[0] = Nlines
        self.Ngaps = Ngaps
        return Nlines

    def GetPnt(self, i : int, increment : int = 1, homogen : bool = False):
        """
            Returns (shift, [x, y])
        """
        shift = 0
        while True:
            
            pnt = self[:(2 + homogen), (i + shift) % self.Nlines[0]]
            if (abs(pnt[0]) > 1e-5 or abs(pnt[1]) > 1e-5):
                return (pnt, shift)

            shift += increment

    def Closest_pnt(self, pnt : np.ndarray, vx_from : int, vx_to : int, minDist = inf, onlyGaps : bool = False):
        """
            Returns (seg_i, closestPnt, minDist)
        """
        seg_i = -1
        closestPnt = np.zeros([2])
        closestPnt_ = np.zeros([2])

        seg_r = np.zeros([2])
        v = np.zeros([2])
        r = np.zeros([2])

        while vx_from < vx_to:
            p1, shift = self.GetPnt(vx_from + 1)
            if (not onlyGaps):
                if (shift != 0):
                    vx_from += (shift + 1)
                    continue
            elif (shift == 0):
                vx_from += 1
                continue
            p0, _ = self.GetPnt(vx_from)

            seg_r[0] = p1[0] - p0[0]
            seg_r[1] = p1[1] - p0[1]

            v[0] = pnt[0] - p0[0]
            v[1] = pnt[1] - p0[1]

            val = np.dot(seg_r, seg_r)
            if (val < 1e-4):
                t = 0.0
            else:
                t = np.dot(v, seg_r) / val
                if (t < 0.0):
                    t = 0.0
                elif (t > 1.0):
                    t = 1.0

            closestPnt_[0] = p0[0] + t * seg_r[0]
            closestPnt_[1] = p0[1] + t * seg_r[1]
            r[0] = pnt[0] - closestPnt_[0]
            r[1] = pnt[1] - closestPnt_[1]
            
            dist = norm(r)
            if (dist < minDist - 1e-9):
                seg_i = vx_from + shift
                closestPnt = closestPnt_.copy()
                minDist = dist

            vx_from += (shift + 1)

        return (seg_i, closestPnt, minDist)

    def Closest_pnt00(self, vx_from : int, vx_to : int, minDist = inf, onlyGaps : bool = False):
        """
            Returns (seg_i, closestPnt, minDist)
        """
        seg_i = -1
        closestPnt = np.zeros([2])
        closestPnt_ = np.zeros([2])

        seg_r = np.zeros([2])

        while vx_from < vx_to:
            p1, shift = self.GetPnt(vx_from + 1)
            if (not onlyGaps):
                if (shift != 0):
                    vx_from += (shift + 1)
                    continue
            elif (shift == 0):
                vx_from += 1
                continue
            p0, _ = self.GetPnt(vx_from)

            seg_r[0] = p1[0] - p0[0]
            seg_r[1] = p1[1] - p0[1]

            val = np.dot(seg_r, seg_r)
            if (val < 1e-4):
                t = 0.0
            else:
                t = -np.dot(p0[:2], seg_r) / val
                if (t < 0.0):
                    t = 0.0
                elif (t > 1.0):
                    t = 1.0

            closestPnt_[0] = p0[0] + t * seg_r[0]
            closestPnt_[1] = p0[1] + t * seg_r[1]
            
            dist = norm(closestPnt_)
            if (dist < minDist - 1e-9):
                seg_i = vx_from + shift
                closestPnt = closestPnt_.copy()
                minDist = dist

            vx_from += (shift + 1)

        return (seg_i, closestPnt, minDist)

    def Check_segment_intersections(self, p0 : np.ndarray, p1 : np.ndarray, vx_from : int, vx_to : int, checkAll : bool = False, num : int = 0, ignoreGaps : bool = False):
        """
            Returns the num of intersections
        """

        if (checkAll):
            px = np.zeros([2])

        seg_r = np.array([(p1[0] - p0[0]) / 2.0, (p1[1] - p0[1]) / 2.0])
        seg_c = p0[:2] + seg_r
        segLen = 2 * norm(seg_r)

        lines_seg_r = np.zeros([2])

        L0 = np.zeros([2]) #axis along segment normal
        L1 = np.zeros([2]) #axis along lines segment normal

        if (abs(seg_r[0]) > 1e-6):
            L0[0] = seg_r[1] / seg_r[0]
            L0[1] = -1.0
        else:
            L0[0] = -1.0
            L0[1] = 0.0

        T = np.zeros([2])

        while vx_from < vx_to:

            p3, shift = self.GetPnt(vx_from + 1)
            if (ignoreGaps and shift != 0):
                vx_from += (shift + 1)
                continue
            p2, _ = self.GetPnt(vx_from)

            lines_seg_r[0] = (p3[0] - p2[0]) / 2.0
            lines_seg_r[1] = (p3[1] - p2[1]) / 2.0

            T[0] = p2[0] + lines_seg_r[0] - seg_c[0]
            T[1] = p2[1] + lines_seg_r[1] - seg_c[1] 

        #L0 along segment normal

            if (abs(np.dot(T, L0)) - abs(np.dot(lines_seg_r, L0)) > -1e-12):
                vx_from += (shift + 1)
                continue

        #L1 along lines_segment normal

            if (abs(lines_seg_r[0]) > 1e-6):
                L1[0] = lines_seg_r[1] / lines_seg_r[0]
                L1[1] = -1.0
            else:
                L1[0] = -1.0
                L1[1] = 0.0

            if (abs(np.dot(T, L1)) - abs(np.dot(seg_r, L1)) > -1e-12):
                vx_from += (shift + 1)
                continue

            if (not checkAll):
                return 1    #do intersect

            if (abs(L0[1]) > 1e-6):
                k1 = -L0[0] / L0[1] #line 1
                if (abs(L1[1]) > 1e-6):
                    k2 = -L1[0] / L1[1]
                    b1 = p0[1] - k1 * p0[0]
                    b2 = p2[1] - k2 * p2[0]
                    px[0] = (b2 - b1) / (k1 - k2)
                    px[1] = k1 * px[0] + b1
                else:
                    px[0] = self.GetPnt(vx_from)[0][0]
                    px[1] = k1 * (p2[0] - p0[0]) + p0[1]
            else:
                px[0] = p0[0] #b1
                px[1] = -L1[0] / L1[1] * (p0[0] - p2[0]) + p2[1] #k2 * b1 + b2, for b2 look double else


            self.checkList[0, num] = vx_from
            self.checkList[1, num] = norm(px - p0[:2]) / segLen
            self.checkList[2, num] = (shift != 0)
        
            num += 1
            vx_from += (shift + 1)
        
        return num

    def Check_segment_intersections00(self, p1 : np.ndarray, vx_from : int, vx_to : int, checkAll : bool = False, num : int = 0, ignoreGaps : bool = False):
        """
            Returns the num of intersections
        """

        if (checkAll):
            px = np.zeros([2])

        seg_r = p1[:2] / 2.0
        segLen = 2 * norm(seg_r)

        lines_seg_r = np.zeros([2])

        L0 = np.zeros([2]) #axis along segment normal
        L1 = np.zeros([2]) #axis along lines segment normal

        if (abs(seg_r[0]) > 1e-6):
            L0[0] = seg_r[1] / seg_r[0]
            L0[1] = -1.0
        else:
            L0[0] = -1.0
            L0[1] = 0.0

        T = np.zeros([2])

        while vx_from < vx_to:

            p3, shift = self.GetPnt(vx_from + 1)
            if (ignoreGaps and shift != 0):
                vx_from += (shift + 1)
                continue
            p2, _ = self.GetPnt(vx_from)

            lines_seg_r[0] = (p3[0] - p2[0]) / 2.0
            lines_seg_r[1] = (p3[1] - p2[1]) / 2.0

            T[0] = p2[0] + lines_seg_r[0] - seg_r[0]
            T[1] = p2[1] + lines_seg_r[1] - seg_r[1] 

        #L0 along segment normal

            if (abs(np.dot(T, L0)) - abs(np.dot(lines_seg_r, L0)) > -1e-12):
                vx_from += (shift + 1)
                continue

        #L1 along lines_segment normal

            if (abs(lines_seg_r[0]) > 1e-6):
                L1[0] = lines_seg_r[1] / lines_seg_r[0]
                L1[1] = -1.0
            else:
                L1[0] = -1.0
                L1[1] = 0.0

            if (abs(np.dot(T, L1)) - abs(np.dot(seg_r, L1)) > 1e-12):
                vx_from += (shift + 1)
                continue

            if (not checkAll):
                return 1    #are intersect

            if (abs(L0[1]) > 1e-6):
                k1 = -L0[0] / L0[1] #line 1
                if (abs(L1[1]) > 1e-6):
                    k2 = -L1[0] / L1[1]
                    px[0] = (p2[1] - k2 * p2[0]) / (k1 - k2)
                    px[1] = k1 * px[0]
                else:
                    px[0] = self.GetPnt(vx_from)[0][0]
                    px[1] = k1 * p2[0]
            else:
                px[0] = 0.0 #b1
                px[1] = L1[0] / L1[1] * p2[0] + p2[1] #k2 * b1 + b2, for b2 look double else

            self.checkList[0, num] = vx_from
            self.checkList[1, num] = norm(px) / segLen
            self.checkList[2, num] = (shift != 0)
        
            num += 1
            vx_from += (shift + 1)
        
        return num

    def Check_if_obb_intersection(self, obb_L2G : np.ndarray, obb_half_length, obb_half_width, vx_from : int, vx_to : int):
        """
            obb - is oriented bounding box on the plane, obb_l2G is 3x3 dimensions.\n
            Returns the first intersected segment index, or -1 if there is no intersections found.
            Gaps are ignored
        """

        #FrontUp radius
        obb_r1 = np.array([obb_L2G[0, 0] * obb_half_length + obb_L2G[0, 1] * obb_half_width, obb_L2G[1, 0] * obb_half_length + obb_L2G[1, 1] * obb_half_width])
        #FrontDown radius
        obb_r2 = np.array([obb_L2G[0, 0] * obb_half_length - obb_L2G[0, 1] * obb_half_width, obb_L2G[1, 0] * obb_half_length - obb_L2G[1, 1] * obb_half_width])
        #OBB is symmetric, so here are only 2 radii out of 4

        lines_seg_r = np.zeros([2])

        L0 = np.zeros([2]) #axis normal to segment
        L1 = np.zeros([2]) #axis along length of obb
        L2 = np.zeros([2]) #axis along width of obb
        L1[:] = obb_L2G[:2, 0]
        L2[:] = obb_L2G[:2, 1]

        T = np.zeros([2])

        while vx_from < vx_to:

            p1, shift = self.GetPnt(vx_from + 1)
            if (shift != 0):
                vx_from += (shift + 1)
                continue
            p0, _ = self.GetPnt(vx_from)

            lines_seg_r[0] = (p1[0] - p0[0]) / 2.0
            lines_seg_r[1] = (p1[1] - p0[1]) / 2.0

            T[0] = p0[0] + lines_seg_r[0] - obb_L2G[0, 2] 
            T[1] = p0[1] + lines_seg_r[1] - obb_L2G[1, 2] 

        #L0 along segment

            if (abs(lines_seg_r[0]) > 1e-5):
                L0[0] = lines_seg_r[1] / lines_seg_r[0]
                L0[1] = -1.0
            else:
                L0[0] = -1.0
                L0[1] = 0

            abs_T_L = abs(np.dot(T, L0))
            # abs_lines_seg_r_L = 0.0
            abs_obb_r1_L = abs(np.dot(obb_r1, L0))
            abs_obb_r2_L = abs(np.dot(obb_r2, L0))

            if (abs_obb_r1_L >= abs_obb_r2_L):
                if (abs_T_L > abs_obb_r1_L):
                    vx_from += 1
                    continue
            elif (abs_T_L > abs_obb_r2_L):
                vx_from += 1
                continue

        #L1 along length of obb

            abs_T_L = abs(np.dot(T, L1))
            abs_lines_seg_r_L = abs(np.dot(lines_seg_r, L1))
            abs_obb_r1_L = abs(np.dot(obb_r1, L1))
            abs_obb_r2_L = abs(np.dot(obb_r2, L1))

            if (abs_obb_r1_L >= abs_obb_r2_L):
                if (abs_T_L > (abs_obb_r1_L + abs_lines_seg_r_L)):
                    vx_from += 1
                    continue
            elif (abs_T_L > (abs_obb_r2_L + abs_lines_seg_r_L)):
                vx_from += 1
                continue

        #L2 along width of obb

            abs_T_L = abs(np.dot(T, L2))
            abs_lines_seg_r_L = abs(np.dot(lines_seg_r, L2))
            abs_obb_r1_L = abs(np.dot(obb_r1, L2))
            abs_obb_r2_L = abs(np.dot(obb_r2, L2))

            if (abs_obb_r1_L >= abs_obb_r2_L):
                if (abs_T_L > (abs_obb_r1_L + abs_lines_seg_r_L)):
                    vx_from += 1
                    continue
            elif (abs_T_L > (abs_obb_r2_L + abs_lines_seg_r_L)):
                vx_from += 1
                continue

            return vx_from    #intersecting with segment vx_from .. vx_from + 1 and maybe others next
        
        return -1   #non-intersecting at all

    def plotLines(self, ax):
                
        v = 0
        while v < self.Nlines[0]:
            
            u = v

            while (v < self.Nlines[0] and (abs(self[0, v]) > 1e-5 or abs(self[1, v]) > 1e-5)):
                v += 1

            if (v > u + 1):
                ax.plot(self[0, u : v], self[1, u : v], linewidth = 3.0, color = 'red')

            if (v < self.Nlines[0] - 1): 
                if (self[0, v] != 0.0 and self[1, v] != 0.0):
                    ax.plot([self[0, v - 1], self[0, v + 1]], [self[1, v - 1], self[1, v + 1]], color = 'black', linewidth = 3.0)
            else:
                ax.plot([self[0, self.Nlines[0] - 1], self[0, 0]], [self[1, self.Nlines[0] - 1], self[1, 0]], color = 'red', linewidth = 3.0)

            v += 1

def Transform_mat(dx,dy,dz,Ox,Oy,Oz,order):
    """ transformation matrix around Ox,Oy,Oz (in rad) and shifts dx,dy,dz
    order > 0: Ox, Oy, Oz, shift
    order < 0: shift, Oz, Oy, Ox 
    order = 0: shift """
    Td = np.array([ \
        [1.0, 0.0, 0.0, dx], \
        [0.0, 1.0, 0.0, dy], \
        [0.0, 0.0, 1.0, dz], \
        [0.0, 0.0, 0.0, 1.0] ])
    if (order == 0):
        return Td
    else:
        TOx = np.array([ \
                [1.0, 0.0, 0.0, 0.0], \
                [0.0, cos(Ox), sin(-Ox), 0.0], \
                [0.0, sin(Ox), cos(Ox), 0.0], \
                [0.0, 0.0, 0.0, 1.0] ])
        TOy = np.array([ \
                [cos(Oy), 0.0, sin(Oy), 0.0], \
                [0.0, 1.0, 0.0, 0.0], \
                [sin(-Oy), 0.0, cos(Oy), 0.0], \
                [0.0, 0.0, 0.0, 1.0] ])
        TOz = np.array([ \
                [cos(Oz), sin(-Oz), 0.0, 0.0], \
                [sin(Oz), cos(Oz), 0.0, 0.0], \
                [0.0, 0.0, 1.0, 0.0], \
                [0.0, 0.0, 0.0, 1.0] ])
        if (order < 0):
            return matmul(matmul(matmul(TOx, TOy), TOz), Td)
        else:
            return matmul(matmul(matmul(Td, TOz), TOy), TOx)

def Transform2D_mat(dx, dy, Oz, order):
    """ transformation matrix around Oz (in rad) and shifts dx,dy
    order > 0: Oz, dx, dy
    order < 0: dx, dy, Oz 
    order = 0: dx, dy """
    if (order > 0):
        return np.array([ \
        [cos(Oz), -sin(Oz), dx], \
        [sin(Oz), cos(Oz), dy], \
        [0.0, 0.0, 1.0] ])
    elif (order < 0):
        return np.array([ \
        [cos(Oz), -sin(Oz), dx * cos(Oz) - dy * sin(Oz)], \
        [sin(Oz), cos(Oz), dx * sin(Oz) + dy * cos(Oz)], \
        [0.0, 0.0, 1.0] ])
    else:
        return np.array([ \
        [1.0, 0.0, dx], \
        [0.0, 1.0, dy], \
        [0.0, 0.0, 1.0] ])

def RotateAroundPnt2D_mat(Oz, pnt):
    return np.array([ \
        [cos(Oz), -sin(Oz), pnt[0] * (1 - cos(Oz)) + pnt[1] * sin(Oz)], \
        [sin(Oz), cos(Oz),  pnt[1] * (1 - cos(Oz)) + pnt[0] * sin(Oz)], \
        [0.0, 0.0, 1.0] ])

def Trilaterate(R, NofAnchors, slope, base, xyz = np.array([0.0, 0.0, 0.0, 1.0]), dur = 0.01):
    """Simple vector descent method.
    xyz - init estimation of the position to be optimized in usual cartesian or homogeneous coordinates
    and it is changing over this func, so send copy if it is unwish.
    Finally xyz becomes an optimized result and this func also returns it
    R (at least 4 x at least NofAnchors):
    % x1 x2 x3 x4 x5...;
    % y1 y2 y3 y4 y5...;
    % z1 z2 z3 z4 z5...;
    % r1 r2 r3 r4 r5...; where
    xi, yi, zi coordinates of anchors (spheres)
    ri - distances to corresponding anchors (spheres)
    NofAnchors - the number of useful anchors in R"""

    NofAnchors_ = 0

    wghts = 1.0 / (1.0 + np.exp( slope * (abs( norm(R[:3, :NofAnchors] - np.reshape(xyz[:3], (3, 1)), axis = 0) - R[3, :NofAnchors] ) - base) ))
    sum = wghts.sum()
    if (sum < 0.01):
        return NofAnchors_
    else:
        wghts /= sum

    for i in range(NofAnchors):
        if (wghts[i] > 0.33 / NofAnchors):
            NofAnchors_ += 1

    rvector = np.zeros([3, NofAnchors])
    t0 = time.time()
    while True:
        delta = np.zeros(3)
        for i in range(NofAnchors):
            rvector[:, i] =  R[:3, i] - xyz[:3]
            delta += rvector[:, i] * (1.0 - R[3, i] / norm(rvector[:, i])) * wghts[i]
        
        if (norm(delta) < 1e-3 or time.time() - t0 > dur):
            return NofAnchors_

        xyz[:3] += delta

def TrilaterateIgnore(R, NofAnchors, ignoreMeases : set, xyz : np.ndarray = np.array([0.0, 0.0, 0.0, 1.0]), dur = 0.01):
    """Simple vector descent method.
    xyz - init position of the optimization in usual cartesian or homogeneous coordinates
    and it is changing over this func, so send copy if it is unwish.
    Finally xyz becomes an optimized result and this func also returns it
    R (at least 4 x at least NofAnchors):
    % x1 x2 x3 x4 x5...;
    % y1 y2 y3 y4 y5...;
    % z1 z2 z3 z4 z5...;
    % r1 r2 r3 r4 r5...; where
    xi, yi, zi coordinates of anchors (spheres)
    ri - distances to corresponding anchors (spheres)
    NofAnchors - the number of useful anchors in R"""

    N = NofAnchors - len(ignoreMeases)

    rvector = np.zeros([3, NofAnchors])
    t0 = time.time()
    while True:
        delta = np.zeros(3)
        for i in range(NofAnchors):
            if (i in ignoreMeases):
                continue
            rvector[:, i] =  R[:3, i] - xyz[:3]
            delta += rvector[:, i] * (1.0 - R[3, i] / norm(rvector[:, i])) / N
        
        if (norm(delta) < 1e-3 or time.time() - t0 > dur):
            return xyz

        xyz[:3] += delta

def TrilaterateWghtd(R, NofAnchors, wghts, xyz : np.ndarray = np.array([0.0, 0.0, 0.0, 1.0]), dur = 0.01):
    """Simple vector descent method.
    xyz - init position of the optimization in usual cartesian or homogeneous coordinates
    and it is changing over this func, so send copy if it is unwish.
    Finally xyz becomes an optimized result and this func also returns it
    R (at least 4 x at least NofAnchors):
    % x1 x2 x3 x4 x5...;
    % y1 y2 y3 y4 y5...;
    % z1 z2 z3 z4 z5...;
    % r1 r2 r3 r4 r5...; where
    xi, yi, zi coordinates of anchors (spheres)
    ri - distances to corresponding anchors (spheres)
    wi - wheights of related measurement
    NofAnchors - the number of useful anchors in R"""

    rvector = np.zeros([3, NofAnchors])
    t0 = time.time()
    while True:
        delta = np.zeros(3)
        for i in range(NofAnchors):
            rvector[:, i] =  R[:3, i] - xyz[:3]
            delta += rvector[:, i] * (1.0 - R[3, i] / norm(rvector[:, i])) * wghts[i]
        
        if (norm(delta) < 1e-3 or time.time() - t0 > dur):
            return xyz

        xyz[:3] += delta

def Trilaterate2(R, NofAnchors, xyz = np.array([0.0, 0.0, 0.0, 1.0]), dur = 0.01):
    """Simple vector descent method. (v.2: more appropriate descent vector but slower, no benefits, no negatives at the exploration)
    xyz - starting position of the optimization in usual cartesian or homogeneous coordinates (if uknown [0, 0, 0] default) 
    and it is changing over this func, so send copy if it is unwish.
    Finally xyz becomes an optimized result and this func also returns it
    R (at least 4 x at least NofAnchors):
    % x1 x2 x3 x4 x5...;
    % y1 y2 y3 y4 y5...;
    % z1 z2 z3 z4 z5...;
    % r1 r2 r3 r4 r5...; where
    xi, yi, zi coordinates of anchors (spheres)
    ri - distances to corresponding anchors (spheres)
    NofAnchors - number of useful anchors in R"""

    rvector = np.zeros([3, NofAnchors])
    rnorm = np.zeros(NofAnchors)
    bias = np.zeros(NofAnchors)
    t0 = time.time()
    while True:
        maxbias = 0.0
        delta = np.zeros(3)
        for i in range(NofAnchors):
            rvector[:, i] =  R[:3, i] - xyz[:3]
            rnorm[i] = norm(rvector[:, i])
            bias[i] = rnorm[i] - R[3, i]
            delta += rvector[:, i] * bias[i] / rnorm[i]
            if math.fabs(bias[i]) > maxbias:
                maxbias = math.fabs(bias[i])
        
        if (norm(bias) < 0.005 or time.time() - t0 > dur):
            return xyz

        xyz[:3] += delta * maxbias / norm(delta)

def Sum_dist2(xyz, R, NofAnchors):
    sumdist2 = 0.0
    for i in range(NofAnchors):
        sumdist2 += (norm(xyz - R[:3, i]) - R[3, i])**2
    return sumdist2

def Trilaterate_old(R, xyz, OptLimitsG, NofAnchors, n):
    """Newton method
    xyz - starting position of optimization in usual cartesian or homogeneous coordinates (the closer, the better) 
    and it is changing over func, so send copy if it is unwish.
    Finally xyz becomes an optimized result and func also returns it
    R (at least 4 x at least NofAnchors):
    % x1 x2 x3 x4 x5...;
    % y1 y2 y3 y4 y5...;
    % z1 z2 z3 z4 z5...;
    % r1 r2 r3 r4 r5...; where
    xi, yi, zi coordinates of anchors (spheres)
    ri - distances to corresponding anchors (spheres)
    OptLimitsG - limits of optimization moves
    NofAnchors - number of useful anchors in R
    n - num of iterations"""

    e = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    h = 0.00001
    H = np.zeros([3, 3])
    G = np.zeros([3])
    dxyz = G

    OptLimitsG /= n

    for _ in range(n):
        r = 0
        while (r < 3):
            G[r] = ( Sum_dist2(xyz[:3] + e[:,r] * h, R, NofAnchors) - Sum_dist2(xyz[:3], R, NofAnchors)) / h
            c = 0
            while (c <= r-1):
                H[r,c] = H[c,r]
                c += 1
            c = r
            while (c < 3):
                H[r,c] = ( Sum_dist2(xyz[:3] + (e[:,r] + e[:,c]) * h, R, NofAnchors) - Sum_dist2(xyz[:3] + e[:,c] * h, R, NofAnchors) ) / h**2 - G[r] / h
                c += 1
            r += 1

        if (norm(G) > 0.003):
            matmul(inv(H),G, dxyz)
            for k in range(3):
                if (abs(dxyz[k]) > OptLimitsG[k]):
                    if (dxyz[k] > 0.0):
                        dxyz[k] = OptLimitsG[k]
                    elif (dxyz[k] < 0.0):
                        dxyz[k] = -OptLimitsG[k]
                    else:
                        dxyz[k] = 0.0
            xyz[:3] -= dxyz[:3]
        else:
            break
        
    return xyz

def TrilaterateAnalytic(R):
    """this trilateration admit non-strict (noisy) case by fading z coord, whereas x and y estimations are accurater
    if the strict case was passed the result is strict too.
    R (at least 4 x at least 4):
    % x1 x2 x3 x4 x5...;
    % y1 y2 y3 y4 y5...;
    % z1 z2 z3 z4 z5...;
    % r1 r2 r3 r4 r5...; where
    xi, yi, zi coordinates of anchors (spheres)
    ri - distances to corresponding anchors (spheres)
    (only the first 4 sph are used)"""

    xyzc = np.array([0.0, 0.0, 0.0, 1.0])
    spheres = np.concatenate((R[:3,:], [[1.0, 1.0, 1.0, 1.0]]), axis = 0)
    rads = R[3,:]

    #transform CS to simple case
    dx = -spheres[0, 0]
    dy = -spheres[1, 0]
    dz = -spheres[2, 0]
    spheres1 = matmul(Transform_mat(dx, dy, dz, 0.0, 0.0, 0.0, 0), spheres)

    Oz = -atan2(spheres1[1, 1], spheres1[0, 1])
    matmul(Transform_mat(0.0, 0.0, 0.0, 0.0, 0.0, Oz, 1), spheres1, spheres1)

    Oy = atan2(spheres1[2, 1], spheres1[0, 1])
    matmul(Transform_mat(0.0, 0.0, 0.0, 0.0, Oy, 0.0, 1), spheres1, spheres1)

    Ox = -atan2(spheres1[2, 2], spheres1[1, 2])
    matmul(Transform_mat(0.0, 0.0, 0.0, Ox, 0.0, 0.0, 1), spheres1, spheres1)

    #trilaterate
    xyzc[0] = (rads[0]**2 - rads[1]**2 + spheres1[0, 1]**2) / 2.0 / spheres1[0, 1]
    xyzc[1] = (rads[0]**2 - rads[2]**2 + spheres1[0, 2]**2 + spheres1[1, 2]**2) / 2.0 / spheres1[1, 2] - xyzc[0] * spheres1[0, 2] / spheres1[1, 2]
    xyzc[2] = rads[0]**2 - xyzc[0]**2 - xyzc[1]**2
    if (xyzc[2] <= 0.0):
        xyzc[2] = 0.0
    else:
        xyzc1 = np.hstack((xyzc[:2], math.sqrt(xyzc[2]), 1.0))
        xyzc2 = np.hstack((xyzc[:2], -xyzc[2], 1.0))
        if (abs(norm(xyzc1 - spheres[:, 3]) - rads[3]) < abs(norm(xyzc2 - spheres[:, 3]) - rads[3])):
            xyzc[2] = xyzc1[2]
        else:
            xyzc[2] = xyzc2[2]

    xyz = matmul(Transform_mat(-dx, -dy, -dz, -Ox, -Oy, -Oz, 1), xyzc)
    #get back to original CS
    return xyz