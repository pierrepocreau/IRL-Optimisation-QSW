import quantumStrategies
from game import Game
from seesaw import SeeSaw
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

def seeSawIteration(seeSaw, QeqFlag):
    '''
    Make a seesaw iteration.
    '''
    maxDif = 0

    # We start by optimizing on rho. Except if QeqFlag is true.
    if not QeqFlag:
            print("Optimisation de rho")
            rho = seeSaw.sdpRho()
            seeSaw.updateRho(rho)
    else:
        print("Optimisation du gain de chaque joueur.")

    for player in range(seeSaw.nbJoueurs):
        print("player {}".format(player))
        playerPOVM = seeSaw.sdpPlayer(player, QeqFlag)
        seeSaw.update(player, playerPOVM)
        maxDif = max(maxDif, seeSaw.lastDif)

    print("QSW {}".format(seeSaw.QSW))
    print("Winrate {}".format(seeSaw.winrate))
    return maxDif

def fullSeeSaw(game, dimension=2, init=None, treshold=1e-6):
    '''
    Seesaw iterations until convergence.
    '''
    seeSaw = SeeSaw(dimension=dimension, game=game, init=init)

    maxDif = np.infty
    iteration = 1
    iterationTreshold = 60

    #Optimisation of QSW with modification of both Rho and POVMs.
    while maxDif >= treshold:
        print("\niteration {}".format(iteration))
        maxDif = seeSawIteration(seeSaw, QeqFlag=False)
        iteration += 1
        #Abort if iteration don't converge fast enough
        if iteration >= iterationTreshold: return 0, seeSaw

    maxDif = np.infty

    #Optimisation of each player's payout, without modification of rho. (equivalent to quantum equlibrium check SDP)
    while maxDif >= treshold:
        print("\niteration {}".format(iteration))
        maxDif = seeSawIteration(seeSaw, QeqFlag=True)
        iteration += 1
        if iteration >= iterationTreshold: return 0, seeSaw

    return seeSaw.QSW, seeSaw

def quantumEqCheck(game, POVMS, rho, threshold, dimension=2):
    """
    SDP to check if a strategy is a quantum equlibrium.
    """
    print("\nQuantum equilibrium check")
    seeSaw = SeeSaw(dimension=dimension, game=game, init=(POVMS, rho))

    maxUpdate = 0
    for player in range(nbPlayers):
        print("player {}".format(player))
        print("current payout {}".format(seeSaw.playersPayout[player]))
        print("Optimization")
        playerPOVM = seeSaw.sdpPlayer(player, Qeq=True)
        maxUpdate = max(maxUpdate, seeSaw.lastDif)
        print(maxUpdate)

    return maxUpdate <= threshold


if __name__ == '__main__':
    nbPlayers = 3
    v0 = 2/3
    v1 = 2 - v0
    dimension = 2
    symmetric=False
    game = Game(nbPlayers, v0, v1, sym=symmetric)
    qsw, seeSaw = fullSeeSaw(game, dimension=dimension)

    print("Test if seesaw strategy is quantum equilibrium: ",
          quantumEqCheck(game, seeSaw.POVM_Dict, seeSaw.rho, threshold=10e-6, dimension=dimension))

    theta = quantumStrategies.optimalTheta(game)
    povms = quantumStrategies.devStratPOVMs(theta, nbPlayers)
    rho = quantumStrategies.devStratRho(theta, nbPlayers)

    print("Test if deviated strategy is quantum equilibrium: ",
          quantumEqCheck(game, povms, rho, threshold=10e-6, dimension=dimension))



