from Operator import *
from itertools import product
import unittest
import numpy as np
from Game import Game
from SDP import SDP

from time import time


class Test(unittest.TestCase):

    def testTemps(self):
        n = 2**5
        A = np.random.rand(n,n)
        B = np.random.rand(n,n)

        a = time()
        C = np.dot(A, B)
        b = time()
        print("temps {}".format(b - a))




if __name__ == "__main__":
    unittest.main()
