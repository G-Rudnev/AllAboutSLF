import numpy as np
import time

class LinesXY(np.ndarray):

    """
        Gapped polyline, pnts are in homogeneous coordinates.\n
        The first and the last pnts should be non-zero both.\n
        Two gaps should be separated by at least one non-gap pnt
    """

    def __new__(cls, N, *args, **kwargs):
        obj = np.zeros([2, N])
        return obj.view(cls)

    def __init__(self, N : int, closed : bool = False):
        self.Nlines = N
        self.closed = closed

    def Fill(self, linesXY_ : np.ndarray, Nlines : int):
        self[:2, :Nlines] = linesXY_[:2, :Nlines]
        self.Nlines = Nlines
        return Nlines

    def GetPnt(self, i : int, increment : int = 1):
        """
            Returns (shift, [x, y])
        """
        shift = 0
        while True:
            
            j = i + shift

            if (j >= self.Nlines or j < 0):
                if (self.closed):
                    j %= self.Nlines
                else:
                    return (None, shift)
                
            pnt = self[:2, j]
            if (abs(pnt[0]) > 0.01 or abs(pnt[1]) > 0.01):
                return (pnt, shift)

            shift += increment

linesXY = LinesXY(4, True)
linesXY.Fill(np.random.rand(2, 4), 4)

print(0.0 < None)

x = 3
y = True
print(x + y)

exit()



