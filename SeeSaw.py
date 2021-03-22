import numpy as np
import cvxpy as cp
from toqitoRandomPovm import random_povm
from Game import Game


class SeeSaw:

    def __init__(self, dimension, game):
        self.dimension = dimension
        self.game = game
        self.nbJoueurs = self.game.nbPlayers

        self.playersPayout = [0 for _ in range(self.nbJoueurs)]
        self.POVM_Dict = self.genPOVMs()
        self.rho = self.genRho()
        self.QSW = 0

    def genPOVMs(self):
        '''
        Initialise each player with random POVMs
        '''
        opDict = {}
        for playerId in range(self.nbJoueurs):
            povms = random_povm(2, 2, 2) #dim = 2, nbInput = 2, nbOutput = 2
            for type in ["0", "1"]:
                for answer in ["0", "1"]:
                    opDict[str(playerId) + answer + type] = povms[:, :, int(type), int(answer)].real

        return opDict

    def genRho(self):
        dim = 2 ** self.nbJoueurs
        #It is not necesseray to initialize rho if it's the first parameter optimised.
        rho = np.zeros(shape=(dim, dim))
        return rho

    def probaPlayer(self, answer, question, playerId, playerPOVM):
        '''
        Calculate the probability p(a|t) with playersId's POVMs being cvxpy vars.
        '''
        IdPOVM = answer[0] + question[0]
        if playerId == 0:
            matrix = playerPOVM[IdPOVM]
        else:
            matrix = self.POVM_Dict["0" + IdPOVM]

        for player in range(1, self.nbJoueurs):
            IdPOVM = answer[player] + question[player]
            if player == playerId:
                matrix = np.kron(matrix, playerPOVM[IdPOVM])
            else:
                matrix = np.kron(matrix, self.POVM_Dict[str(player) + IdPOVM])

        matrix = cp.bmat(matrix)
        trace = cp.trace(self.rho @ matrix)
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

        return cp.trace(rho @ cp.bmat(matrix))


    def update(self, playerId, playerPOVM):
        '''
        Update playersId's POVMs after convex optimisation.
        '''
        dist = 0
        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                newOp = [[playerPOVM[answer + type][0][0].value, playerPOVM[answer + type][0][1].value], [playerPOVM[answer + type][1][0].value, playerPOVM[answer + type][1][1].value]]
                dist = max(dist, np.linalg.norm(self.POVM_Dict[str(playerId) + answer + type] - newOp))
                self.POVM_Dict[str(playerId) + answer + type] = np.array(newOp)

        print("Max diff between old POVMs and new {}".format(dist))


    def sdpPlayer(self, playerId, Qeq):
        '''
        Build and solve optimisation for playerId
        With parameters, we could build it only once.
        '''
        constraints = []
        varDict = {}

        for type in ["0", "1"]:
            for answer in ["0", "1"]:
                varMatrix = cp.Variable((2, 2), PSD=True)
                #We must create a matrix by hand, cp.Variabel((2,2)) can't be used as first arguement of cp.kron(a, b) or np.kron(a, b)
                var = [[varMatrix[0, 0], varMatrix[0, 1]], [varMatrix[1, 0], varMatrix[1, 1]]]
                varDict[answer + type] = np.array(var)

        constraints += [cp.bmat(varDict["00"] + varDict["10"]) == np.eye(2)]
        constraints += [cp.bmat(varDict["01"] + varDict["11"]) == np.eye(2)]

        socialWelfare = cp.Constant(0)
        playerPayout = cp.Constant(0)
        winrate = cp.Constant(0)

        for question in self.game.questions():
            for answer in self.game.validAnswerIt(question):
                proba = self.probaPlayer(answer, question, playerId, varDict)
                socialWelfare += 1/4 * self.game.answerPayout(answer) * proba
                playerPayout += 1/4 * self.game.playerPayout(answer, playerId) * proba
                winrate += 1/4 * proba

        if Qeq:
            sdp = cp.Problem(cp.Maximize(playerPayout), constraints)
        else:
            sdp = cp.Problem(cp.Maximize(socialWelfare), constraints)

        sdp.solve(solver=cp.MOSEK, verbose=False)
        self.QSW = socialWelfare.value
        self.winrate = winrate.value
        self.update(playerId, varDict)
        print("player payout {} updateDiff {}".format(playerPayout.value, playerPayout.value - self.playersPayout[playerId]))
        self.playersPayout[playerId] = playerPayout.value

    def sdpRho(self):
        '''
        Build and solve optimisation for rho
        With parameters, we could build it only once.
        '''
        constraints = []
        n = 2**self.nbJoueurs
        rho = cp.Variable((n, n), PSD=True)
        constraints += [cp.trace(rho) == 1]

        socialWelfaire = cp.Constant(0)
        winrate = cp.Constant(0)

        for question in self.game.questions():
            for answer in self.game.validAnswerIt(question):
                proba = self.probaRho(answer, question, rho)
                socialWelfaire += 1/4 * self.game.answerPayout(answer) * proba
                winrate  += 1/4 * proba

        sdp = cp.Problem(cp.Maximize(socialWelfaire), constraints)
        sdp.solve(solver=cp.MOSEK, verbose=False)

        #Update rho
        self.rho = rho.value