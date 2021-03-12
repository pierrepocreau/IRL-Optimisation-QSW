import numpy as np
import matplotlib.pyplot as plt
from Game import Game
from SDP import SDP
import time
import cvxpy as cp

def graph(points):
    operatorsP1 = [0, 1, 2]
    operatorsP2 = [0, 3, 4]
    operatorsP3 = [0, 5, 6]
    operatorsP4 = [0, 7, 8]
    operatorsP5 = [0, 9, 10]

    P3 = [operatorsP1, operatorsP2, operatorsP3]
    P5 = [operatorsP1, operatorsP2, operatorsP3, operatorsP4, operatorsP5]
    x = np.linspace(0, 1, points)

    nbPlayers = 3
    v1 = 1
    paramV0 = cp.Parameter()
    game = Game(nbPlayers, paramV0, v1, P3)
    prob = SDP(game)

    QSW_Nash = []
    QSW_notNash = []

    print("Sans contrainte de Nash")
    for idx, v0 in enumerate(x):
        print("iteration {}".format(idx))
        qswGraphState = (v0 + v1) / 2

        paramV0.value = v0
        qsw = prob.optimize(verbose=False, warmStart=False)
        QSW_notNash.append(qsw - qswGraphState)

    print("Avec contrainte de Nash")
    prob.nashEquilibriumConstraint()
    prob.updateProb()

    for idx, v0 in enumerate(x):
        print("iteration {}".format(idx))
        qswGraphState = (v0 + v1) / 2

        paramV0.value = v0
        qsw = prob.optimize(verbose=True, warmStart=False)
        QSW_Nash.append(qsw - qswGraphState)

    plt.plot(x, QSW_Nash, label="QSW + Nash constraint")
    plt.plot(x, QSW_notNash, label="QSW without Nash constraint")
    plt.xlabel("V0/V1")
    plt.ylabel("QSW - QSW(GraphState)")
    plt.legend(loc="upper right")

    plt.show()

start = time.time()
graph(100)
end = time.time()
print("time {}".format(end - start))