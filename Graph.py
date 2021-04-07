import numpy as np
import cvxpy as cp
from Game import Game
from SDP import SDP
from mainSeeSaw import fullSeeSaw, printPOVMS, graphStatePOVMS
import matplotlib.pyplot as plt
from toqito.state_metrics import fidelity
import devStrat
import os

graphStateVec = 1 / np.sqrt(8) * np.array([1, 1, 1, -1, 1, -1, -1, -1])
graphState = np.outer(graphStateVec, graphStateVec)

def readFile(file):
    f = open(file, "r")
    data = []
    for element in f.read().split("\n"):
        try:
            data.append(float(element))
        except:
            if element != "":
                print(element)
                data.append(list(element))
    print(data)
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

    x = np.linspace(0, 1, points)

    v1 = 1
    paramV0 = cp.Parameter()
    game = Game(nbPlayers, paramV0, v1, sym)
    prob = SDP(game, P)


    QSW_GraphState = []
    QSW_Nash = []
    QSW_NotNash = []
    QSW_dev = []
    SW_classical = []


    #Saving in a file is not needed since it's fast.
    print("GraphState & deviated strat & classical strat")
    for idx, v0 in enumerate(x):
        print("iteration {}".format(idx))
        qswGraphState = (v0 + v1) / 2
        QSW_GraphState.append(qswGraphState)
        a = 7/12 + 1/6*v0
        SW_classical.append(a)
        QSW_dev.append(devStrat.QSW(v0, v1, devStrat.optimalTheta(v0, v1, nbPlayers), nbPlayers))


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
        if nbPlayers == 3:
            rhoMat = np.load('data/{}Players_{}Points_SeeSaw_psiComponents.npy'.format(nbPlayers, points), allow_pickle = True)

    except:
        print("SeeSaw")
        init = None
        rhoMat = []
        QSW_SeeSaw = []
        Winrate_SeeSaw = []

        if os.path.exists('data/{}Players_{}Points_SeeSaw.txt'.format(nbPlayers, points)):
            os.remove('data/{}Players_{}Points_SeeSaw.txt'.format(nbPlayers, points))

        if os.path.exists('data/{}Players_{}Points_SeeSaw_Winrate.txt'.format(nbPlayers, points)):
            os.remove('data/{}Players_{}Points_SeeSaw_Winrate.txt'.format(nbPlayers, points))

        for it, v0 in enumerate(reversed(x)):
            print("\nGlobal Iteration {}".format(it))
            maxQsw, winrate = 0, 0

            bestPOVMS = None
            bestRho = None
            bestfid = 0

            #Initialisation with graphstate strat
            #implemented for 3 players (need the expression of the graphState for 5 players)
            if nbPlayers == 3:
                bestPOVMS = graphStatePOVMS(nbPlayers)
                bestRho = graphState
                init = (bestPOVMS, bestRho)

            nbRepeat = seeSawRepeatLow * (v0 < treshold) + seeSawRepeatHigh * (v0 >= treshold)

            for r in range(nbRepeat):
                print("nbRepeat {}".format(r))
                qsw, seeSaw = fullSeeSaw(nbPlayers, v0, v1, init=init, dimension=dimension)
                maxQsw = max(maxQsw, qsw)
                print(np.trace(np.dot(seeSaw.rho, seeSaw.rho)))

                # If it's the best result we encoutered yet for this v0's value, we save the strat and it's results.
                if maxQsw == qsw:
                    winrate = seeSaw.winrate
                    bestPOVMS = seeSaw.POVM_Dict
                    bestRho = seeSaw.rho
                    bestSeeSaw = seeSaw
                    bestfid = max(bestfid, fidelity(bestRho, graphState))

            QSW_SeeSaw.append(maxQsw)
            Winrate_SeeSaw.append(winrate)
            init = (bestPOVMS, seeSaw.genRho()) #If we keep old rho, we often (always ?) stay on the same equilibrium.
            printPOVMS(bestSeeSaw)

            #Need the graphstate expression for 5 players
            if nbPlayers == 3:
                #Orthogonal of |psi> where bestRho = |psi><psi|
                direction = graphStateVec - np.dot(graphStateVec, np.dot(graphState, bestRho))
                #Calculation of psi from the orthogonal
                psi = (graphStateVec - direction) / (np.sqrt(np.dot(np.dot(graphStateVec, bestRho), graphStateVec)))
                rhoMat.append(psi)

                print("psi: ", psi)
                print("Fidelity rho & graphstate", bestfid)

        with open('data/{}Players_{}Points_SeeSaw.txt'.format(nbPlayers, points), 'w') as f:
            for item in QSW_SeeSaw:
                f.write("%s\n" % item)

        with open('data/{}Players_{}Points_SeeSaw_Winrate.txt'.format(nbPlayers, points), 'w') as f:
            for item in Winrate_SeeSaw:
                f.write("%s\n" % item)

        if nbPlayers == 3:
            rhoMat = np.array(rhoMat).transpose()
            np.save('data/{}Players_{}Points_SeeSaw_psiComponents'.format(nbPlayers, points), rhoMat)

    fig, axs = plt.subplots(2, constrained_layout = True, figsize=(10, 20))
    fig.suptitle("Graph for {} players with {} points".format(nbPlayers, points))

    axs[0].plot(x, QSW_GraphState, label="GraphState")
    axs[0].plot(x, QSW_Nash, label="HierarchieNash")
    axs[0].plot(x, QSW_NotNash, label="HierarchieNotNash")
    axs[0].plot(x, list(reversed(QSW_SeeSaw)), label="SeeSaw")
    axs[0].plot(x, list(reversed(Winrate_SeeSaw)), label="Winrate Seesaw")
    axs[0].plot(x, QSW_dev, label="stratégie deviée")
    axs[0].plot(x, SW_classical, label="Welfare two players answers 1 and one player Not")

    axs[0].set_title("Quantum social welfare")
    axs[0].set_xlabel("V0/V1")
    axs[0].set_ylabel("QSW")
    axs[0].set_ylim([0, 1])
    axs[0].legend(loc="upper right")

    if nbPlayers == 3:
        axs[1].plot(x, list(reversed(rhoMat[0])), label="|000>")
        axs[1].plot(x, list(reversed(rhoMat[1])), label="|100> (|010> |001>)")
        axs[1].plot(x, list(reversed(rhoMat[2])), label="|010>")
        axs[1].plot(x, list(reversed(rhoMat[3])), label="|110> (|101> |011>)")
        axs[1].plot(x, list(reversed(rhoMat[4])), label="|001>")
        axs[1].plot(x, list(reversed(rhoMat[5])), label="|101>")
        axs[1].plot(x, list(reversed(rhoMat[6])), label="|011>")
        axs[1].plot(x, list(reversed(rhoMat[7])), label="|111>")
        axs[1].set_title("state components")
        axs[1].set_xlabel("V0/V1")
        axs[1].set_ylabel("coefficient")
        axs[1].set_ylim([-0.5, 0.5])
        axs[1].legend(loc="upper right")

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


