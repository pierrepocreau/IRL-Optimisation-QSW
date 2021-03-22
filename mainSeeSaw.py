from Game import Game
from SeeSaw import SeeSaw
import numpy as np

def printPOVMS(seeSaw):
    for id in range(seeSaw.game.nbPlayers):
        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                print("Player " + str(id) + " Operator " + answer + type)
                print(seeSaw.POVM_Dict[str(id) + answer + type])
                print("\n")

def seeSawIteration(seeSaw, QeqFlag):
    if not QeqFlag:
        print("Optimisation de rho")
        seeSaw.sdpRho()
    else:
        print("Optimisation du gain de chaque joueur.")
    for player in range(seeSaw.game.nbPlayers):
        print("player {}".format(player))
        test = seeSaw.sdpPlayer(player, Qeq)
    print("QSW {}".format(seeSaw.QSW))
    print("Winrate {}".format(seeSaw.winrate))


game = Game(3, 2/3, 1, False)
seeSaw = SeeSaw(3, game)

print("POVMs initiaux")
printPOVMS(seeSaw)

nbIterations = 20
for i in range(nbIterations):
    print("\niteration {}".format(i))
    Qeq = (i >= 13 and i <= 20)
    seeSawIteration(seeSaw, Qeq)
print("Etat final")


print(seeSaw.rho)
printPOVMS(seeSaw)

