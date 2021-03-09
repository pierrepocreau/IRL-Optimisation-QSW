from Game import Game
from SDP import SDP

operatorsP1 = [0, 1, 2]
operatorsP2 = [0, 3, 4]
operatorsP3 = [0, 5, 6]
operatorsP4 = [0, 7, 8]
operatorsP5 = [0, 9, 10]

P3 = [operatorsP1, operatorsP2, operatorsP3]
P5 = [operatorsP1, operatorsP2, operatorsP3, operatorsP4, operatorsP5]

nbPlayers = 3
v0 = 2/3
v1 = 1
game = Game(nbPlayers, v0, v1, P3) #To solve the 5 player version, change nbPlayers and P3 to P5

prob = SDP(game)
prob.projectorConstraints()
prob.nashEquilibriumConstraint()
qsw = prob.optimize(verbose=True)
print("QSW = {}".format(qsw))

