from Game import Game
from SeeSaw import SeeSaw
import numpy as np

def printPOVMS(seeSaw):
    for id in range(seeSaw.game.nbPlayers):
        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                print("Player " + str(id) + " Operator " + answer + type)
                print(seeSaw.POVM_Dict[str(id) + answer + type])
                print("\n carré ")
                print(np.dot(seeSaw.POVM_Dict[str(id) + answer + type] , seeSaw.POVM_Dict[str(id) + answer + type]))
                print("\n")

def graphStatePOVMS(nbPlayers):
    POVMS_Dict = {}
    for id in range(nbPlayers):
        POVMS_Dict[str(id) + "00"] = np.array([[1, 0], [0, 0]])
        POVMS_Dict[str(id) + "10"] = np.array([[0, 0], [0, 1]])
        POVMS_Dict[str(id) + "01"] = np.array([[0.5, 0.5], [0.5, 0.5]])
        POVMS_Dict[str(id) + "11"] = np.array([[0.5, -0.5], [-0.5, 0.5]])
    return POVMS_Dict

def graphState(nbPlayers):
    """
    Return the density matrix associated to the graphState for 3 or 5 players.
    """
    assert(nbPlayers == 3 or nbPlayers == 5)
    if nbPlayers == 3:
        graphStateVec = 1 / np.sqrt(8) * np.array([1, 1, 1, -1, 1, -1, -1, -1])
    elif nbPlayers == 5:
        graphStateVec = 1 / np.sqrt(2 ** 5) * np.array([1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1,
                                                        1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1])

    return np.outer(graphStateVec, graphStateVec)

def seeSawIteration(seeSaw, QeqFlag, init=False):
    maxDif = 0
    if not QeqFlag:
        if not init: #If rho is initialized, we don't start by optimising it
            print("Optimisation de rho")
            rho = seeSaw.sdpRho()
            seeSaw.updateRho(rho)
    else:
        print("Optimisation du gain de chaque joueur.")

    #optimOrder = list(range(seeSaw.nbJoueurs))
    #random.shuffle(optimOrder)
    optimOrder = range(seeSaw.nbJoueurs)

    for player in optimOrder:
        print("player {}".format(player))
        playerPOVM = seeSaw.sdpPlayer(player, QeqFlag)
        seeSaw.update(player, playerPOVM)
        maxDif = max(maxDif, seeSaw.lastDif)

    print("QSW {}".format(seeSaw.QSW))
    print("Winrate {}".format(seeSaw.winrate))
    return maxDif

def fullSeeSaw(nbJoueurs, v0, v1, init=None, sym=False, treshold=10e-6, dimension=2):
    game = Game(nbJoueurs, v0,v1, sym)
    seeSaw = SeeSaw(dimension=dimension, game=game, init=init)

    maxDif = 2
    iteration = 1
    while maxDif >= treshold:
        #Decomment if we change Rho
        #initFlag = (init is not None) & (iteration == 1)
        initFlag = False & iteration == 1
        print("\niteration {}".format(iteration))
        maxDif = seeSawIteration(seeSaw, QeqFlag=False, init=initFlag)
        iteration += 1

        #Abort if iteration don't converge fast
        if iteration >= 100: return 0, seeSaw

    maxDif = 2
    while maxDif >= treshold:
        print("\niteration {}".format(iteration))
        maxDif = seeSawIteration(seeSaw, QeqFlag=True, init=False)
        iteration += 1

        if iteration >= 200: return 0, seeSaw


    return seeSaw.QSW, seeSaw

def quantumEqCheck(nbPlayers, v0, v1, POVMS, rho, threshold, dimension=2):
    print("\nQuantum equilibrium check")
    game = Game(nbPlayers, v0, v1, sym=False)
    seeSaw = SeeSaw(dimension=dimension, game=game, init=(POVMS, rho))

    maxUpdate = 0
    for player in range(nbPlayers):
        print("player {}".format(player))
        print("current payout {}".format(seeSaw.playersPayout[player]))
        print("Optimization")
        playerPOVM = seeSaw.sdpPlayer(player, Qeq=True)
        maxUpdate = max(maxUpdate, seeSaw.lastDif)

    #seeSawIteration(seeSaw, QeqFlag=True)
    return maxUpdate <= threshold


if __name__ == '__main__':
    nbPlayers = 5
    v0 = 0.14
    v1 = 1
    dimension = 2
    qsw, seeSaw = fullSeeSaw(nbPlayers, v0, v1, dimension=dimension)
    print(quantumEqCheck(nbPlayers, v0, v1, seeSaw.POVM_Dict, seeSaw.rho, threshold=10e-6, dimension=dimension))
