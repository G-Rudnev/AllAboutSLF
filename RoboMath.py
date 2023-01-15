import time
import math

import numpy as np
from numpy import matmul as matmul
from numpy.linalg import norm as norm
from numpy.linalg import inv as inv

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
                [0.0, math.cos(Ox), math.sin(-Ox), 0.0], \
                [0.0, math.sin(Ox), math.cos(Ox), 0.0], \
                [0.0, 0.0, 0.0, 1.0] ])
        TOy = np.array([ \
                [math.cos(Oy), 0.0, math.sin(Oy), 0.0], \
                [0.0, 1.0, 0.0, 0.0], \
                [math.sin(-Oy), 0.0, math.cos(Oy), 0.0], \
                [0.0, 0.0, 0.0, 1.0] ])
        TOz = np.array([ \
                [math.cos(Oz), math.sin(-Oz), 0.0, 0.0], \
                [math.sin(Oz), math.cos(Oz), 0.0, 0.0], \
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
        [math.cos(Oz), -math.sin(Oz), dx], \
        [math.sin(Oz), math.cos(Oz), dy], \
        [0.0, 0.0, 1.0] ])
    elif (order < 0):
        return np.array([ \
        [math.cos(Oz), -math.sin(Oz), dx * math.cos(Oz) - dy * math.sin(Oz)], \
        [math.sin(Oz), math.cos(Oz), dx * math.sin(Oz) + dy * math.cos(Oz)], \
        [0.0, 0.0, 1.0] ])
    else:
        return np.array([ \
        [1.0, 0.0, dx], \
        [0.0, 1.0, dy], \
        [0.0, 0.0, 1.0] ])

def RotateAroundPnt2D_mat(Oz, pnt):
    return np.array([ \
        [math.cos(Oz), -math.sin(Oz), pnt[0] * (1 - math.cos(Oz)) + pnt[1] * math.sin(Oz)], \
        [math.sin(Oz), math.cos(Oz),  pnt[1] * (1 - math.cos(Oz)) + pnt[0] * math.sin(Oz)], \
        [0.0, 0.0, 1.0] ])

def Sum_dist2(xyz, R, NofAnchors):
    sumdist2 = 0.0
    for i in range(NofAnchors):
        sumdist2 += (norm(xyz - R[:3, i]) - R[3, i])**2
    return sumdist2

def Trilaterate(R, NofAnchors, xyz = np.array([0.0, 0.0, 0.0, 1.0]), dur = 0.01):
    """Simple vector descent method.
    xyz - starting position of the optimization in the usual cartesian or homogeneous coordinates (if unknown, [0, 0, 0] default) 
    and it is changing over this func, so sending copy if it is unwish.
    Finally xyz becomes an optimized result and this func also returns it
    R (at least 4 x at least NofAnchors):
    % x1 x2 x3 x4 x5...;
    % y1 y2 y3 y4 y5...;
    % z1 z2 z3 z4 z5...;
    % r1 r2 r3 r4 r5...; where
    xi, yi, zi coordinates of anchors (spheres)
    ri - distances to corresponding anchors (spheres)
    NofAnchors - the number of useful anchors in R"""

    rvector = np.zeros([3, NofAnchors])
    t0 = time.time()
    while True:
        delta = np.zeros(3)
        for i in range(NofAnchors):
            rvector[:, i] =  R[:3, i] - xyz[:3]
            delta += rvector[:, i] * (1.0 - R[3, i] / norm(rvector[:, i])) / NofAnchors
        
        if (norm(delta) < 0.001 or time.time() - t0 > dur):
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

    Oz = -math.atan2(spheres1[1, 1], spheres1[0, 1])
    matmul(Transform_mat(0.0, 0.0, 0.0, 0.0, 0.0, Oz, 1), spheres1, spheres1)

    Oy = math.atan2(spheres1[2, 1], spheres1[0, 1])
    matmul(Transform_mat(0.0, 0.0, 0.0, 0.0, Oy, 0.0, 1), spheres1, spheres1)

    Ox = -math.atan2(spheres1[2, 2], spheres1[1, 2])
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