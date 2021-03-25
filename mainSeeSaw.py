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
        seeSaw.sdpRho()
    else:
        print("Optimisation du gain de chaque joueur.")

    #optimOrder = list(range(seeSaw.nbJoueurs))
    #random.shuffle(optimOrder)
    optimOrder = range(seeSaw.nbJoueurs)

    for player in optimOrder:
        print("player {}".format(player))
        seeSaw.sdpPlayer(player, QeqFlag)
        maxDif = max(maxDif, seeSaw.lastDif)

    print("QSW {}".format(seeSaw.QSW))
    print("Winrate {}".format(seeSaw.winrate))
    return maxDif

def fullSeeSaw(nbJoueurs, v0, v1, sym=False, treshold=10E-6):
    game = Game(nbJoueurs, v0,v1, sym)
    seeSaw = SeeSaw(nbJoueurs, game)

    maxDif = 2
    iteration = 1
    while maxDif >= treshold:
        print("\niteration {}".format(iteration))
        maxDif = seeSawIteration(seeSaw, False)
        iteration += 1

        if iteration >= 30: return 0

    maxDif = 2
    while maxDif >= treshold:
        print("\niteration {}".format(iteration))
        maxDif = seeSawIteration(seeSaw, True)
        iteration += 1

        if iteration >= 60: return 0


    return seeSaw.QSW


#fullSeeSaw(5, 2/3, 1)
