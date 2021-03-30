import numpy as np
import cvxpy as cp
from Game import Game
from SDP import SDP
from mainSeeSaw import fullSeeSaw, printPOVMS, graphStatePOVMS
import matplotlib.pyplot as plt
import os

def readFile(file):
    f = open(file, "r")
    data = []
    for element in f.read().split("\n"):
        try:
            data.append(float(element))
        except: pass
    return data


def graph(nbPlayers, sym, points, seeSawRepeatLow = 10, seeSawRepeatHigh = 3, treshold=0.4, dimension=2):

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
    Winrate_SeeSaw = []
    test = []

    try :
        QSW_GraphState = readFile('data/{}Players_{}Points_GraphState.txt'.format(nbPlayers, points))
    except:
        print("GraphState")
        for idx, v0 in enumerate(x):
            print("iteration {}".format(idx))
            qswGraphState = (v0 + v1) / 2
            QSW_GraphState.append(qswGraphState)
            a = 7/12 + 1/6*v0
            test.append(a)

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
        Winrate_SeeSaw = readFile('data/{}Players_{}Points_SeeSaw_Winrate.txt'.format(nbPlayers, points))


    except:
        print("SeeSaw")
        init = None

        for it, v0 in enumerate(reversed(x)):
            print("\nGlobal Iteration {}".format(it))
            maxQsw, winrate = 0, 0

            bestPOVMS = None
            bestRho = None

            if nbPlayers == 3:
                bestPOVMS = graphStatePOVMS(nbPlayers)
                graphState = 1 / np.sqrt(8) * np.array([1, 1, 1, -1, 1, -1, -1, -1])
                bestRho = np.outer(graphState, graphState)
                init = (bestPOVMS, bestRho)

            nbRepeat = seeSawRepeatLow * (v0 < treshold) + seeSawRepeatHigh * (v0 >= treshold)

            for r in range(nbRepeat):
                print("nbRepeat {}".format(r))
                qsw, seeSaw = fullSeeSaw(nbPlayers, v0, v1, init=init, dimension=dimension)
                maxQsw = max(maxQsw, qsw)
                print(np.trace(np.dot(seeSaw.rho, seeSaw.rho)))

                if maxQsw == qsw:
                    winrate = seeSaw.winrate
                    bestPOVMS = seeSaw.POVM_Dict
                    bestRho = seeSaw.rho
                    bestSeeSaw = seeSaw


            QSW_SeeSaw.append(maxQsw)
            Winrate_SeeSaw.append(winrate)
            init = (bestPOVMS, seeSaw.genRho()) #If we keep old rho, we often (always ?) stay on the same equilibrium.
            printPOVMS(bestSeeSaw)
            print(bestRho)

        with open('data/{}Players_{}Points_SeeSaw.txt'.format(nbPlayers, points), 'w') as f:
            for item in QSW_SeeSaw:
                f.write("%s\n" % item)

        with open('data/{}Players_{}Points_SeeSaw_Winrate.txt'.format(nbPlayers, points), 'w') as f:
            for item in Winrate_SeeSaw:
                f.write("%s\n" % item)

    plt.plot(x, QSW_GraphState, label="GraphState")
    plt.plot(x, QSW_Nash, label="HierarchieNash")
    plt.plot(x, QSW_NotNash, label="HierarchieNotNash")
    plt.plot(x, list(reversed(QSW_SeeSaw)), label="SeeSaw")
    plt.plot(x, list(reversed(Winrate_SeeSaw)), label="Winrate Seesaw")
    #plt.plot(x, test, label="Welfare two players answers 1 and one player Not")

    plt.xlabel("V0/V1")
    plt.ylabel("QSW")
    plt.legend(loc="upper right")
    plt.title("QSW graph for {} players with {} points".format(nbPlayers, points))
    plt.show()

if __name__ == '__main__':
    nbPlayers = 3
    sym=False #Sym for 5 players
    points = 25
    seeSawRepeatLow = 5
    seeSawRepeatHigh = 5
    treshold = 0.33
    dimension = 2 #Only change dimension used in seeSaw, not on the Hierarchie.

    graph(nbPlayers, sym, points, seeSawRepeatLow, seeSawRepeatHigh, treshold, dimension=dimension)


