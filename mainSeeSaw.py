from Game import Game
from SeeSaw import SeeSaw
import numpy as np
import random

def printPOVMS(seeSaw):
    for id in range(seeSaw.game.nbPlayers):
        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                print("Player " + str(id) + " Operator " + answer + type)
                print(seeSaw.POVM_Dict[str(id) + answer + type])
                print("\n")

def seeSawIteration(seeSaw, QeqFlag):
    maxDif = 0
    if not QeqFlag:
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

def fullSeeSaw(nbJoueurs, v0, v1, sym=False, treshold=10e-6):
    game = Game(nbJoueurs, v0,v1, sym)
    seeSaw = SeeSaw(nbJoueurs, game)

    maxDif = 2
    iteration = 1
    while maxDif >= treshold:
        print("\niteration {}".format(iteration))
        maxDif = seeSawIteration(seeSaw, False)
        iteration += 1

        if iteration >= 30: return 0, seeSaw

    maxDif = 2
    while maxDif >= treshold:
        print("\niteration {}".format(iteration))
        maxDif = seeSawIteration(seeSaw, True)
        iteration += 1

        if iteration >= 60: return 0, seeSaw


    return seeSaw.QSW, seeSaw


def quantumEqCheck(nbPlayers, v0, v1, POVMS, rho, threshold):
    print("\nQuantum equilibrium check")
    game = Game(nbPlayers, v0, v1, sym=False)
    seeSaw = SeeSaw(dimension=2, game=game)
    seeSaw.rho = rho
    seeSaw.POVM_Dict = POVMS

    maxUpdate = 0
    for player in range(nbPlayers):
        print("player {}".format(player))
        seeSaw.currentPayout(player)
        playerPOVM = seeSaw.sdpPlayer(player, Qeq=True)
        maxUpdate = max(maxUpdate, seeSaw.lastDif)

    #seeSawIteration(seeSaw, QeqFlag=True)
    return maxUpdate <= threshold


if __name__ == '__main__':
    nbPlayers = 5
    v0 = 2/3
    v1 = 1
    qsw, seeSaw = fullSeeSaw(nbPlayers, v0, v1)
    print(quantumEqCheck(nbPlayers, v0, v1, seeSaw.POVM_Dict, seeSaw.rho, threshold=10e-6))
