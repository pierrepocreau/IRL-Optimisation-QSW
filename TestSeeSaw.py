from Operator import *
from itertools import product
import unittest
import numpy as np
from Game import Game
from SDP import SDP
from SeeSaw import graph

from time import time
import cvxpy as cp


class Test(unittest.TestCase):

    def testTemps(self):
        n = 2**5
        A = np.random.rand(n,n)
        B = np.random.rand(n,n)
        C = cp.Variable((n,n))

        D = [[cp.Variable() for _ in range(n)] for _ in range(n)]
        t1 = time()
        np.dot(A, B)
        t2 = time()
        A @ C
        t3 = time()
        A @ D
        t4 = time()
        D = cp.bmat(D)
        A @ D
        t5 = time()

        print("temps multiplication classique {} multiplication Variable(n,n) cvxpy {} multiplication matrices 'fait main' {} avec bmat {}".format(t2 - t1, t3 - t2, t4 - t3, t5 - t4))

    def testGrpah(self):
        qsw = graph(25)
        with open('QSW_25Points_SeeSaw.txt', 'w') as f:
            for item in qsw:
                f.write("%s\n" % item)



if __name__ == "__main__":
    unittest.main()
