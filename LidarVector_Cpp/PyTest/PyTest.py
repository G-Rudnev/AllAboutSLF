
import time
import numpy as np
import lidarVector
from importlib import reload

N = 10000000
pntsXY = np.ones([3, N])
pntsPhi = np.ones([N]) * 1.5
linesXY = np.zeros([3, N])

pntsXY1 = np.ones([3, N])
pntsPhi1 = np.ones([N]) * 1.5
linesXY1 = np.zeros([3, N])

pntsXY2 = np.ones([3, N])
pntsPhi2 = np.ones([N]) * 1.5
linesXY2 = np.zeros([3, N])

id0 = lidarVector.init(pntsXY, pntsPhi, linesXY, N, (0.0, ))
id1 = lidarVector.init(pntsXY1, pntsPhi1, linesXY1, N, (0.0, ))
id2 = lidarVector.init(pntsXY2, pntsPhi2, linesXY2, N, (0.0, ))

t0 = time.time()
linesXY[0, :] = np.exp(pntsXY[0, :] * pntsXY[0, :] * pntsPhi)
linesXY[1, :] = np.exp(pntsXY[1, :] * pntsXY[1, :] * pntsPhi)
print('PY calc time:', time.time() - t0)

t0 = time.time()
lidarVector.pushPnts(id0)
lidarVector.synchronize(id0)
print('CPP Push only id0 time:', time.time() - t0)

t0 = time.time()
lidarVector.calcLines(id0)  #если по данному id не идет работа, вызов не блокирует
lidarVector.synchronize(id0)
print('CPP Calc id0 time:', time.time() - t0)

print('lines beg:')
print(linesXY[:, :1])
print('lines end:')
print(linesXY[:, N - 1:])

print('__NEXT DATA__')
pntsXY[:] = 2.0 #обновили данные

t0 = time.time()
linesXY[0, :] = np.exp(pntsXY[0, :] * pntsXY[0, :] * pntsPhi)
linesXY[1, :] = np.exp(pntsXY[1, :] * pntsXY[1, :] * pntsPhi)
print('PY Calc id0 time:', time.time() - t0)

t0 = time.time()
lidarVector.pushPnts(id0) #если по данному id не идет работа, вызов не блокирует
lidarVector.pushPnts(id1)
lidarVector.synchronize(id0)
print('CPP Push id0 and id1 async (no sync for id1) time:', time.time() - t0) #он ускоряется при следующих вызовах push, видимо, кэширует

t0 = time.time()
lidarVector.calcLines(id0)
lidarVector.synchronize(id0)
print('CPP Calc id0 time:', time.time() - t0)

print('lines beg:')
print(linesXY[:, :1])
print('lines end:')
print(linesXY[:, N - 1:])

print('__NEXT DATA__')
pntsXY[:] = 3.0 #обновили данные

t0 = time.time()
lidarVector.pushPnts(id0)
lidarVector.calcLines(id0)
lidarVector.pushPnts(id1)
lidarVector.calcLines(id1)
lidarVector.synchronize(id0)
lidarVector.synchronize(id1)
print('CPP Push and Calc id0, id1 async time:', time.time() - t0)

t0 = time.time()
lidarVector.pushPnts(id0)
lidarVector.calcLines(id0)
lidarVector.pushPnts(id1)
lidarVector.calcLines(id1)
lidarVector.pushPnts(id2)
lidarVector.calcLines(id2)
lidarVector.synchronize(id0)
lidarVector.synchronize(id1)
lidarVector.synchronize(id2)
print('CPP Push and Calc id0, id1, id2 async time:', time.time() - t0)

print('lines beg:')
print(linesXY[:, :1])
print('lines end:')
print(linesXY[:, N - 1:])
