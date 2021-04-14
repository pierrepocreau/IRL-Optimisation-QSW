import numpy as np
import cvxpy as cp
from game import Game
from hierarchie import Hierarchie
from seesawUtils import fullSeeSaw, printPOVMS, graphStatePOVMS, graphState
import matplotlib.pyplot as plt
from toqito.state_metrics import fidelity
import devStrat
import os


def readFile(file):
    '''
    Utility function, to load saved txt file.
    '''
    f = open(file, "r")
    data = []
    for element in f.read().split("\n"):
        try:
            data.append(float(element))
        except:
            if element != "":
                print(element)
                data.append(list(element))
    return data


def graph(nbPlayers, sym, points, seeSawRepeatLow = 10, seeSawRepeatHigh = 3, treshold=0.4, dimension=2):

    operatorsP1 = [0, 1, 2]
    operatorsP2 = [0, 3, 4]
    operatorsP3 = [0, 5, 6]
    operatorsP4 = [0, 7, 8]
    operatorsP5 = [0, 9, 10]
    P3 = [operatorsP1, operatorsP2, operatorsP3]
    P5 = [operatorsP1, operatorsP2, operatorsP3, operatorsP4, operatorsP5]
    if nbPlayers == 5: P = P5
    else: P = P3
    x = np.linspace(0, 1, points, sym)

    v1 = 1
    paramV0 = cp.Parameter()
    game = Game(nbPlayers, paramV0, v1, sym)
    prob = Hierarchie(game, P)

    print("GraphState & deviated strat & classical strat")
    QSW_GraphState = []
    SW_classical = []
    QSW_dev = []

    for idx, v0 in enumerate(x):
        print("iteration {}".format(idx))

        #QSW of graphstate Strat
        qswGraphState = (v0 + v1) / 2
        QSW_GraphState.append(qswGraphState)

        #SW for a classical strat (I don't know if it works for 5 players too)
        #Pour 5 rajouter meilleur (papier)
        a3 = 7/12 + 1/6*v0
        SW_classical.append(a3)

        #QSW for deviated strat
        dev = devStrat.QSW(v0, v1, devStrat.optimalTheta(v0, v1, nbPlayers), nbPlayers)
        QSW_dev.append(dev)

    try:
        QSW_NotNash = readFile('data/{}Players_{}Points_Sym{}_HierarchieNoNash.txt'.format(nbPlayers, points, sym))
        print("Chargement hierarchie sans contrainte de Nash")

    except:
        print("Hierarchie Sans contrainte de Nash")
        QSW_NotNash = []
        for idx, v0 in enumerate(x):
            print("iteration {}".format(idx))
            paramV0.value = v0
            qsw = prob.optimize(verbose=False, warmStart=True, solver="SCS")
            QSW_NotNash.append(qsw)

        with open('data/{}Players_{}Points_Sym{}_HierarchieNoNash.txt'.format(nbPlayers, points, sym), 'w') as f:
            for item in QSW_NotNash:
                f.write("%s\n" % item)

    try:
        QSW_Nash = readFile('data/{}Players_{}Points_Sym{}_HierarchieNash.txt'.format(nbPlayers, points, sym))
        print("Chargement hierarchie avec contrainte de Nash")

    except:
        print("Hierarchie avec contrainte de Nash")
        QSW_Nash = []

        prob.setNashEqConstraints()

        for idx, v0 in enumerate(x):
            print("iteration {}".format(idx))
            paramV0.value = v0
            qsw = prob.optimize(verbose=False, warmStart=True, solver="SCS")
            QSW_Nash.append(qsw)

        with open('data/{}Players_{}Points_Sym{}_HierarchieNash.txt'.format(nbPlayers, points, sym), 'w') as f:
            for item in QSW_Nash:
                f.write("%s\n" % item)

    try:
        QSW_SeeSaw = readFile('data/{}Players_{}Points_Sym{}_SeeSaw.txt'.format(nbPlayers, points, sym))
        Winrate_SeeSaw = readFile('data/{}Players_{}Points_Sym{}_SeeSaw_Winrate.txt'.format(nbPlayers, points, sym))
        print("Chargement SeeSaw")


    except:
        print("SeeSaw")
        graphStateMatrix = graphState(nbPlayers)
        init = (graphStatePOVMS(nbPlayers), graphStateMatrix)

        QSW_SeeSaw = []
        Winrate_SeeSaw = []


        #If one of them exist, we remove the file.
        if os.path.exists('data/{}Players_{}Points_Sym{}_SeeSaw.txt'.format(nbPlayers, points, sym)):
            os.remove('data/{}Players_{}Points_Sym{}_SeeSaw.txt'.format(nbPlayers, points, sym))

        if os.path.exists('data/{}Players_{}Points_Sym{}_SeeSaw_Winrate.txt'.format(nbPlayers, points, sym)):
            os.remove('data/{}Players_{}Points_Sym{}_SeeSaw_Winrate.txt'.format(nbPlayers, points, sym))

        for it, v0 in enumerate(reversed(x)):
            print("\nGlobal Iteration {}".format(it))
            maxQsw = 0

            nbRepeat = seeSawRepeatLow * (v0 < treshold) + seeSawRepeatHigh * (v0 >= treshold)
            #On pourrait couper si on est proche de la borne de la hierarchie

            for r in range(nbRepeat):
                print("nbRepeat {}".format(r))
                qsw, seeSaw = fullSeeSaw(nbPlayers, v0, v1, init=init, dimension=dimension)
                maxQsw = max(maxQsw, qsw)

                # If it's the best result we encoutered yet for this v0's value, we save the strategy.
                if maxQsw == qsw:
                    bestSeeSaw = seeSaw


            QSW_SeeSaw.append(maxQsw)
            Winrate_SeeSaw.append(bestSeeSaw.winrate)
            init = (bestSeeSaw.POVM_Dict, seeSaw.genRho()) #If we keep old rho, we often (always ?) stay on the same equilibrium.
            printPOVMS(bestSeeSaw)
            print("Trace of rho squared:", np.trace(np.dot(bestSeeSaw.rho, bestSeeSaw.rho)))
            print("Fidelity with graphState", fidelity(bestSeeSaw.rho, graphStateMatrix))

        with open('data/{}Players_{}Points_Sym{}_SeeSaw.txt'.format(nbPlayers, points, sym), 'w') as f:
            for item in QSW_SeeSaw:
                f.write("%s\n" % item)

        with open('data/{}Players_{}Points_Sym{}_SeeSaw_Winrate.txt'.format(nbPlayers, points, sym), 'w') as f:
            for item in Winrate_SeeSaw:
                f.write("%s\n" % item)

    fig, axs = plt.subplots(1, constrained_layout = True, figsize=(10, 10))
    fig.suptitle("Graph for {} players with {} points, sym: {}".format(nbPlayers, points, sym))

    axs.plot(x, QSW_GraphState, label="GraphState")
    axs.plot(x, QSW_Nash, label="HierarchieNash")
    axs.plot(x, QSW_NotNash, label="HierarchieNotNash")
    axs.plot(x, list(reversed(QSW_SeeSaw)), label="SeeSaw")
    axs.plot(x, list(reversed(Winrate_SeeSaw)), label="Winrate Seesaw")
    axs.plot(x, QSW_dev, label="stratégie deviée")
    axs.plot(x, SW_classical, label="Welfare two players answers 1 and one player Not")

    axs.set_title("Quantum social welfare")
    axs.set_xlabel("V0/V1")
    axs.set_ylabel("QSW")
    axs.set_ylim([0, 1])
    axs.legend(loc="upper right")

    plt.show()

if __name__ == '__main__':
    nbPlayers = 3
    sym=False #Sym for 5 players
    points = 7
    seeSawRepeatLow = 5
    seeSawRepeatHigh = 5
    treshold = 0.33
    dimension = 2 #Only change dimension used in seeSaw, not on the Hierarchie.

    graph(nbPlayers, sym, points, seeSawRepeatLow, seeSawRepeatHigh, treshold, dimension=dimension)


