from Game import Game
from hierarchie import SDP
import cvxpy as cp

operatorsP1 = [0, 1, 2]
operatorsP2 = [0, 3, 4]
operatorsP3 = [0, 5, 6]
operatorsP4 = [0, 7, 8]
operatorsP5 = [0, 9, 10]

P3 = [operatorsP1, operatorsP2, operatorsP3]
P5 = [operatorsP1, operatorsP2, operatorsP3, operatorsP4, operatorsP5]

nbPlayers = 5

v0 = cp.Parameter()
v1 = cp.Parameter()

game = Game(nbPlayers, v0, v1, sym=False) #To solve the 5 player version, change nbPlayers and P3 to P5

prob = SDP(game, P5)
prob.nashEquilibriumConstraint()
v0.value = 2/3
v1.value = 1
qsw = prob.optimize(verbose=True, warmStart=False)
print("QSW = {}".format(qsw))


