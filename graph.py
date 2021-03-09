import numpy as np
import matplotlib.pyplot as plt
from Game import Game
from SDP import SDP

def graph(points):
    operatorsP1 = [0, 1, 2]
    operatorsP2 = [0, 3, 4]
    operatorsP3 = [0, 5, 6]
    P3 = [operatorsP1, operatorsP2, operatorsP3]

    x = np.linspace(0, 1, points)
    nbPlayers = 3
    v1 = 1

    QSW_Nash = []
    QSW_notNash = []

    for idx, v0 in enumerate(x):
        print("iteration {}".format(idx))
        qswGraphState = (v0 + v1) / 2

        game = Game(nbPlayers, v0, v1, P3)
        prob = SDP(game)
        prob.projectorConstraints()
        prob.nashEquilibriumConstraint()
        qsw = prob.optimize(False)
        QSW_Nash.append(qsw - qswGraphState)

        game = Game(nbPlayers, v0, v1, P3)
        prob = SDP(game)
        prob.projectorConstraints()
        qsw = prob.optimize(False)
        QSW_notNash.append(qsw - qswGraphState)


    plt.plot(x, QSW_Nash, label="QSW + Nash constraint")
    plt.plot(x, QSW_notNash, label="QSW without Nash constraint")
    plt.xlabel("V0/V1")
    plt.ylabel("QSW - QSW(GraphState)")
    plt.legend(loc="upper right")


    plt.show()

graph(200)