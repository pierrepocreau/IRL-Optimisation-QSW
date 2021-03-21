import numpy as np
import cvxpy as cp
from Game import Game
from toqitoRandomPovm import random_povm
from time import time
import matplotlib.pyplot as plt

class SeeSaw:

    def __init__(self, dimension, game):
        self.dimension = dimension
        self.game = game
        self.nbJoueurs = self.game.nbPlayers

        self.game = game

        self.playersOps = self.operators()
        self.rho = self.genRho()
        self.QSW = 0

    def probaPlayer(self, answer, question, playerId, vars):
        '''
        Calculate the probability p(a|t) with playersId's POVMS being SDP vars.
        '''
        a = time()
        first = answer[0] + question[0]
        if playerId == 0:
            matrix = vars[first]
        else:
            matrix = self.playersOps["0" + first]

        for player in range(1, self.nbJoueurs):
            opId = answer[player] + question[player]
            if player == playerId:
                matrix = np.kron(matrix, vars[opId])
            else:
                matrix = np.kron(matrix, self.playersOps[str(player) + opId])

        b = time()
        trace = np.trace(self.rho @ matrix)
        c = time()
        #print("temps produit tensoriel {} multiplication {}".format(b -a, c - b))
        return trace

    def probaRho(self, answer, question, rho):
        '''
        Calculate the probability p(a|t) with rho being SDP variables.
        '''

        firstOp = answer[0] + question[0]
        matrix = self.playersOps["0" + firstOp]

        for player in range(1, self.nbJoueurs):
            opId = answer[player] + question[player]
            matrix = np.kron(matrix, self.playersOps[str(player) + opId])

        return cp.trace(rho @ matrix)


    def update(self, playerId, vars):
        '''
        Update playersId's POVMs after sdp optimisation.
        '''
        dist = 0
        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                newOp = [[vars[answer + type][0][0].value, vars[answer + type][0][1].value], [vars[answer + type][1][0].value, vars[answer + type][1][1].value]]
                dist = max(dist, np.linalg.norm(self.playersOps[str(playerId) + answer + type] - newOp))
                self.playersOps[str(playerId) + answer + type] = np.array(newOp)
        print("Max diff between old POVMs and new {}".format(dist))

    def operators(self):

        opDict = {}
        for playerId in range(self.nbJoueurs):
            #Je comprend pas vraiment ces paramètres, et pourquoi on obtient 4 matrices.
            povms = random_povm(2, 2, 2) #dim = 2, nbInput = 2, nbOutput = 2

            for type in ["0", "1"]:
                for answer in ["0", "1"]:
                    #povms[:, :, int(type), int(answer)][0, 1] = 0
                    #povms[:, :, int(type), int(answer)][1, 0] = 0

                    opDict[str(playerId) + answer + type] = povms[:, :, int(type), int(answer)].real

                print(opDict[str(playerId) + "0" + type] + opDict[str(playerId) + "1" + type])

        return opDict

    def genRho(self):
        dim = 2 ** self.nbJoueurs
        #It is not necesseray to initialize rho if it's the first parameter optimised.
        rho = np.zeros(shape=(dim, dim))
        return rho

    def corrQConstraints(self, playerId, vars):
        '''
        Not full corrQ constraints, it only check for the player being optimised, not for other.
        '''
        constraints = []
        payout = cp.Constant(0)
        for question in self.game.questions():
            for validAnswer in self.game.validAnswerIt(question):
                payout += self.game.playerPayout(validAnswer, playerId) * self.probaPlayer(validAnswer, question, playerId, vars)
        payout *= self.game.questionDistribution

        # Payout for strat which diverge from advice
        for type in ['0', '1']:
            for noti in ['0', '1']:
                payoutNot = cp.Constant(0)
                for question in self.game.questions():
                    stillValid = lambda answer: question[playerId] != type or answer[playerId] != noti
                    nowValid = lambda answer: question[playerId] == type and answer[playerId] == noti

                    for validAnswer in filter(stillValid, self.game.validAnswerIt(question)):
                        payoutNot += self.game.playerPayout(validAnswer, playerId) * self.probaPlayer(validAnswer,question, playerId, vars)

                    for validAnswer in filter(nowValid, self.game.wrongAnswerIt(question)):
                        payoutNot += self.game.notPlayerPayout(validAnswer, playerId) * self.probaPlayer(validAnswer,question, playerId, vars)

                payoutNot *= self.game.questionDistribution
                constraints += [payout >= payoutNot]

        return constraints

    def sdpPlayer(self, playerId, Qeq):
        constraints = []
        varDict = {}

        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                #We must create a matrix by hand, cp.Variabel((2,2)) can't be used as first arguement of cp.kron(a, b) or np.kron(a, b)
                varMatrix = cp.Variable((2, 2), PSD=True)
                var = [[varMatrix[0, 0], varMatrix[0, 1]], [varMatrix[1, 0], varMatrix[1, 1]]]
                varDict[answer + type] = np.array(var)

        constraints += [cp.bmat(varDict["00"] + varDict["10"]) == np.eye(2)]
        constraints += [cp.bmat(varDict["01"] + varDict["11"]) == np.eye(2)]

        if False:
            constraints += self.corrQConstraints(playerId, varDict)

        objectif = cp.Constant(0)
        winrate = cp.Constant(0)
        playerPayout = cp.Constant(0)
        for question in self.game.questions():
            for answer in self.game.validAnswerIt(question):
                proba = self.probaPlayer(answer, question, playerId, varDict)
                objectif += 1/4 * self.game.answerPayout(answer) * proba
                playerPayout += 1/4 * self.game.playerPayout(answer, playerId) * proba
                winrate += 1/4 * proba

        if Qeq:
            sdp = cp.Problem(cp.Maximize(playerPayout), constraints)
        else:
            sdp = cp.Problem(cp.Maximize(objectif), constraints)


        sdp.solve(solver=cp.SCS, verbose=False)
        #print("QSW: " + str(sdp.value))
        #print("Winrate {} Payout {} player payout {}".format(str(winrate.value), str(objectif.value), str(playerPayout.value)))
        self.QSW = objectif.value

        #To print each p(a|t) after optim.
        #for question in game.questions():
        #    for answer in game.validAnswerIt(question):
        #        print("answer {} question {} proba {}".format(answer, question, self.probaPlayer(answer, question, playerId, varDict).value))

        self.update(playerId, varDict)

    def sdpRho(self):
        constraints = []
        n = 2**self.nbJoueurs
        rho = cp.Variable((n, n), PSD=True)
        constraints += [cp.trace(rho) == 1]


        objectif = cp.Constant(0)
        winrate = cp.Constant(0)
        for question in self.game.questions():
            for answer in self.game.validAnswerIt(question):
                proba = self.probaRho(answer, question, rho)
                objectif += 1/4 * self.game.answerPayout(answer) * proba
                winrate  += 1/4 * proba

        sdp = cp.Problem(cp.Maximize(objectif), constraints)
        sdp.solve(solver=cp.MOSEK, verbose=False)
        #print("Winrate {} QSW {} ".format(str(winrate.value), str(sdp.value)))

        #Update rho
        self.rho = rho.value

def graph(points):
    x = np.linspace(0, 1, points)
    QSW = []

    v1 = 1
    nbPlayers = 3

    nbIterations = 30
    nbRepeat = 5
    print(len(x))
    for it, v0 in enumerate(x):
        print("\nIteration {}".format(it))
        maxQsw = 0

        for r in range(nbRepeat):
            print("nbRepeat {}".format(r))
            game = Game(nbPlayers, v0, v1, sym=False)
            seeSaw = SeeSaw(nbPlayers, game)
            for i in range(nbIterations):
                print("Playerit {}".format(i))
                Qeq = i >= nbIterations - 10
                if not Qeq: seeSaw.sdpRho()
                for player in range(game.nbPlayers):
                    seeSaw.sdpPlayer(player, Qeq)
                print("QSW {}".format(seeSaw.QSW))

            maxQsw = max(maxQsw, seeSaw.QSW)
        print("Max QSW {}".format(maxQsw))
        QSW.append(maxQsw)
    plt.plot(x, QSW)
    plt.show()
    return QSW







if __name__ == "__main__":
    graphState = (1/np.sqrt(8)) * np.array([1, 1, 1, -1, 1, -1, -1, -1])

    game = Game(3, 2/3, 1, False)
    seeSaw = SeeSaw(3, game)
    #rho = np.kron(graphState, graphState).reshape((8,8))
    #seeSaw.rho = rho

    print("POVMs initiaux")

    for id in range(game.nbPlayers):
        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                print("Player " + str(id) + " Operator " + answer + type)
                print(seeSaw.playersOps[str(id) + answer + type])
                print("\n")


    nbIterations = 40
    for i in range(nbIterations):
        print("\niteration {}".format(i))
        print("Optimisation de rho")
        Qeq = (i >=  13 and i <= 20) or (i >=  33)
        if not Qeq: seeSaw.sdpRho()
        for player in range(game.nbPlayers):
            print("player {}".format(player))
            sdp = seeSaw.sdpPlayer(player, Qeq)
        print("QSW {}".format(seeSaw.QSW))

    print("Etat final")
    print(seeSaw.rho)

    for id in range(game.nbPlayers):
        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                print("Player " + str(id) + " Operator " + answer + type)
                print(seeSaw.playersOps[str(id) + answer + type])
                print("\n")
