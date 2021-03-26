import numpy as np
import cvxpy as cp
from Game import Game
from SDP import SDP
from mainSeeSaw import fullSeeSaw
import matplotlib.pyplot as plt

def readFile(file):
    f = open(file, "r")
    data = []
    for element in f.read().split("\n"):
        try:
            data.append(float(element))
        except: pass
    return data


def graph(nbPlayers, sym, points, seeSawRepeatLow = 10, seeSawRepeatHigh = 3, treshold=0.4):

    operatorsP1 = [0, 1, 2]
    operatorsP2 = [0, 3, 4]
    operatorsP3 = [0, 5, 6]
    operatorsP4 = [0, 7, 8]
    operatorsP5 = [0, 9, 10]
    operatorsP5 = [0, 9, 10]

    P3 = [operatorsP1, operatorsP2, operatorsP3]
    P5 = [operatorsP1, operatorsP2, operatorsP3, operatorsP4, operatorsP5]

    if nbPlayers == 5: P = P5
    else: P = P3

    x = np.linspace(0, 1, points)

    v1 = 1
    paramV0 = cp.Parameter()
    game = Game(nbPlayers, paramV0, v1, sym)
    prob = SDP(game, P)


    QSW_GraphState = []
    QSW_Nash = []
    QSW_NotNash = []
    QSW_SeeSaw = []

    try :
        QSW_GraphState = readFile('data/{}Players_{}Points_GraphState.txt'.format(nbPlayers, points))
    except:
        print("GraphState")
        for idx, v0 in enumerate(x):
            print("iteration {}".format(idx))
            qswGraphState = (v0 + v1) / 2
            QSW_GraphState.append(qswGraphState)

            with open('data/{}Players_{}Points_GraphState.txt'.format(nbPlayers, points), 'w') as f:
                for item in QSW_GraphState:
                    f.write("%s\n" % item)


    try:

        QSW_NotNash = readFile('data/{}Players_{}Points_HierarchieNoNash.txt'.format(nbPlayers, points))

    except:
        print("Hierarchie Sans contrainte de Nash")
        for idx, v0 in enumerate(x):
            print("iteration {}".format(idx))
            paramV0.value = v0
            qsw = prob.optimize(verbose=False, warmStart=True)
            QSW_NotNash.append(qsw)

        with open('data/{}Players_{}Points_HierarchieNoNash.txt'.format(nbPlayers, points), 'w') as f:
            for item in QSW_NotNash:
                f.write("%s\n" % item)


    try:
        QSW_Nash = readFile('data/{}Players_{}Points_HierarchieNash.txt'.format(nbPlayers, points))
    except:
        print("Hierarchie avec contrainte de Nash")
        prob.nashEquilibriumConstraint()

        for idx, v0 in enumerate(x):
            print("iteration {}".format(idx))
            paramV0.value = v0
            qsw = prob.optimize(verbose=False, warmStart=True)
            QSW_Nash.append(qsw)

        with open('data/{}Players_{}Points_HierarchieNash.txt'.format(nbPlayers, points), 'w') as f:
            for item in QSW_Nash:
                f.write("%s\n" % item)

    try:
        QSW_SeeSaw = readFile('data/{}Players_{}Points_SeeSaw.txt'.format(nbPlayers, points))

    except:
        print("SeeSaw")

        for it, v0 in enumerate(x):
            print("\nGlobal Iteration {}".format(it))
            maxQsw = 0
            nbRepeat = seeSawRepeatLow * (v0 < treshold) + seeSawRepeatHigh * (v0 >= treshold)

            for r in range(nbRepeat):
                print("nbRepeat {}".format(r))
                qsw, seeSaw = fullSeeSaw(nbPlayers, v0, v1)
                maxQsw = max(maxQsw, qsw)

            QSW_SeeSaw.append(maxQsw)

        with open('data/{}Players_{}Points_SeeSaw.txt'.format(nbPlayers, points), 'w') as f:
            for item in QSW_SeeSaw:
                f.write("%s\n" % item)

    plt.plot(x, QSW_GraphState, label="GraphState")
    plt.plot(x, QSW_Nash, label="SDPNash")
    plt.plot(x, QSW_NotNash, label="SDPNotNash")
    plt.plot(x, QSW_SeeSaw, label="SeeSaw")
    plt.xlabel("V0/V1")
    plt.ylabel("QSW")
    plt.legend(loc="upper right")
    plt.title("QSW graph for {} players with {} points".format(nbPlayers, points))
    plt.show()

if __name__ == '__main__':
    nbPlayers = 3
    sym=False #Sym for 5 players
    points = 25
    seeSawRepeatLow = 10
    seeSawRepeatHigh = 3
    treshold = 0.4

    graph(nbPlayers, sym, points, seeSawRepeatLow, seeSawRepeatHigh, treshold)


