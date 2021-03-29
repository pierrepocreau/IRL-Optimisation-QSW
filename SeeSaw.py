import numpy as np
import cvxpy as cp
from toqitoRandomPovm import random_povm
from Game import Game
from toqito.channels import partial_trace
from time import time


class SeeSaw:

    def __init__(self, dimension, game, init=None):
        self.dimension = dimension
        self.game = game
        self.nbJoueurs = self.game.nbPlayers
        self.playersPayout = [0 for _ in range(self.nbJoueurs)]

        if init is None:
            self.POVM_Dict = self.genPOVMs()
            self.rho = self.genRho()

        else:
            self.POVM_Dict = init[0]
            self.rho = init[1]
            for playerId in range(self.nbJoueurs):
                self.currentPayout(playerId)

        self.QSW = 0
        self.lastDif = 10 # >0

    def currentPayout(self, playerId):
        playerPayout = 0
        for question in self.game.questions():
            for answer in self.game.validAnswerIt(question):
                proba = self.proba(answer, question)
                playerPayout += self.game.questionDistribution * self.game.playerPayout(answer, playerId) * proba

        self.playersPayout[playerId] = playerPayout

    def proba(self, answer, question):
        '''
        Calculate the probability p(a|t) with current povms
        '''

        IdPOVM = answer[0] + question[0]
        matrix = self.POVM_Dict["0" + IdPOVM]

        for player in range(1, self.nbJoueurs):
            IdPOVM = answer[player] + question[player]
            matrix = np.kron(matrix, self.POVM_Dict[str(player) + IdPOVM])

        matrix = np.dot(self.rho, matrix)
        return np.trace(matrix)

    def genPOVMs(self):
        '''
        Initialise each player with random POVMs
        '''
        opDict = {}
        for playerId in range(self.nbJoueurs):
            for type in ["0", "1"]:
                for answer in ["0", "1"]:
                    povms = random_povm(self.dimension, 2, self.dimension)  # dim = 2, nbInput = 2, nbOutput = 2

                    opDict[str(playerId) + answer + type] = povms[:, :, int(type), int(answer)].real

        return opDict

    def genRho(self):
        dim = self.dimension ** self.nbJoueurs
        #It is not necesseray to initialize rho if it's the first parameter optimised.
        rho = np.zeros(shape=(dim, dim))
        return rho

    def probaPlayer(self, answer, question, playerId, playerPOVM):
        '''
        Calculate the probability p(a|t) with playersId's POVMs being cvxpy vars.
        '''

        IdPOVM = answer[0] + question[0]
        if playerId == 0:
            matrix = np.eye(self.dimension)
        else:
            matrix = self.POVM_Dict["0" + IdPOVM]

        for player in range(1, self.nbJoueurs):
            IdPOVM = answer[player] + question[player]
            if player == playerId:
                matrix = np.kron(matrix, np.eye(self.dimension))
            else:
                matrix = np.kron(matrix, self.POVM_Dict[str(player) + IdPOVM])

        matrix = self.rho @ matrix
        matrix = partial_trace(matrix, [player+1 for player in range(self.nbJoueurs) if player != playerId], [self.dimension for _ in range(self.nbJoueurs)])
        trace = cp.trace(matrix @ playerPOVM[answer[playerId] + question[playerId]])
        return trace

    def probaRho(self, answer, question, rho):
        '''
        Calculate the probability p(a|t) with rho being cvxpy variable.
        '''

        IdPOVM = answer[0] + question[0]
        matrix = self.POVM_Dict["0" + IdPOVM]

        for player in range(1, self.nbJoueurs):
            opId = answer[player] + question[player]
            matrix = np.kron(matrix, self.POVM_Dict[str(player) + opId])

        return cp.trace(rho @ matrix)


    def update(self, playerId, playerPOVM):
        '''
        Update playersId's POVMs after convex optimisation.
        '''
        dist = 0
        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                dist = max(dist, np.linalg.norm(self.POVM_Dict[str(playerId) + answer + type] - playerPOVM[answer + type].value))
                self.POVM_Dict[str(playerId) + answer + type] = playerPOVM[answer + type].value

        #print("Max diff between old POVMs and new {}".format(dist))


    def sdpPlayer(self, playerId, Qeq):
        '''
        Build and solve optimisation for playerId
        With parameters, we could build it only once.
        '''
        constraints = []
        varDict = {}

        for type in ["0", "1"]:
            for answer in [str(a) for a in range(self.dimension)]:
                varMatrix = cp.Variable((self.dimension, self.dimension), PSD=True)
                #We must create a matrix by hand, cp.Variabel((2,2)) can't be used as first arguement of cp.kron(a, b) or np.kron(a, b)
                #var = [[varMatrix[0, 0], varMatrix[0, 1]], [varMatrix[1, 0], varMatrix[1, 1]]]
                varDict[answer + type] = varMatrix

        constraint0 = varDict["00"]
        constraint1 = varDict["01"]

        for a in range(1, self.dimension):
            constraint0 += varDict[str(a) + "0"]
            constraint1 += varDict[str(a) + "1"]


        constraints += [cp.bmat(constraint0) == np.eye(self.dimension)]
        constraints += [cp.bmat(constraint1) == np.eye(self.dimension)]

        socialWelfare = cp.Constant(0)
        playerPayout = cp.Constant(0)
        winrate = cp.Constant(0)

        for question in self.game.questions():
            for answer in self.game.validAnswerIt(question):
                proba = self.probaPlayer(answer, question, playerId, varDict)
                socialWelfare += self.game.questionDistribution * self.game.answerPayout(answer) * proba
                playerPayout += self.game.questionDistribution * self.game.playerPayout(answer, playerId) * proba
                winrate += self.game.questionDistribution * proba


        if Qeq:
            sdp = cp.Problem(cp.Maximize(playerPayout), constraints)
        else:
            sdp = cp.Problem(cp.Maximize(socialWelfare), constraints)

        sdp.solve(solver=cp.MOSEK, verbose=False)
        self.QSW = socialWelfare.value
        self.winrate = winrate.value
        self.lastDif = np.abs(playerPayout.value - self.playersPayout[playerId])
        print("player payout {} updateDiff {}".format(playerPayout.value, self.lastDif))

        self.playersPayout[playerId] = playerPayout.value
        return varDict


    def sdpRho(self):
        '''
        Build and solve optimisation for rho
        With parameters, we could build it only once.
        '''
        constraints = []
        n = self.dimension**self.nbJoueurs
        rho = cp.Variable((n, n), PSD=True)
        constraints += [cp.trace(rho) == 1]

        socialWelfaire = cp.Constant(0)
        winrate = cp.Constant(0)

        for question in self.game.questions():
            for answer in self.game.validAnswerIt(question):
                proba = self.probaRho(answer, question, rho)
                socialWelfaire += self.game.questionDistribution * self.game.answerPayout(answer) * proba
                winrate += self.game.questionDistribution * proba

        sdp = cp.Problem(cp.Maximize(socialWelfaire), constraints)
        sdp.solve(solver=cp.MOSEK, verbose=False)

        return rho


    def updateRho(self, rho):
        self.rho = rho.value
