from Operator import *
from itertools import product
import unittest
import numpy as np
from Game import Game
from SeeSaw import SeeSaw
import matplotlib.pyplot as plt

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
        qsw = self.graph(25)
        with open('QSW_25Points_0-04_SeeSaw.txt', 'w') as f:
            for item in qsw:
                f.write("%s\n" % item)

    def graph(self, points):
        x = np.linspace(0, 1, points)
        QSW = []

        v1 = 1
        nbPlayers = 3

        nbIterations = 20
        nbRepeat = 3
        print(len(x))
        for it, v0 in enumerate(x):
            print("\nIteration {}".format(it))
            maxQsw = 0

            for r in range(nbRepeat):
                print("nbRepeat {}".format(r))
                game = Game(nbPlayers, v0, v1, sym=False)
                seeSaw = SeeSaw(nbPlayers, game)
                for i in range(nbIterations):

                    print("\nPlayerit {}".format(i))
                    Qeq = i >= nbIterations - 8
                    seeSaw.sdpRho()
                    for player in range(game.nbPlayers):
                        seeSaw.sdpPlayer(player, Qeq)
                    print("QSW {}".format(seeSaw.QSW))
                    print("Winrate {}".format(seeSaw.winrate))

                maxQsw = max(maxQsw, seeSaw.QSW)
            print("Max QSW {}".format(maxQsw))
            QSW.append(maxQsw)
        plt.plot(x, QSW)
        plt.show()
        return QSW


if __name__ == "__main__":
    unittest.main()
