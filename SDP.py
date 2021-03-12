import cvxpy as cp
import numpy as np
from Operator import Variable

class SDP:

    def __init__(self, game):
        self.S = game.S
        self.n = len(game.S)
        self.game = game

        self.variableDict = {}
        self.variablePosition = {}

        self.constraints = []
        self.X = cp.bmat(self.init_variables())
        self.constraints += [self.X >> 0] #SDP
        self.constraints += [self.X[0][0] == 1] #Normalization

        self.objectifFunc = self.objectifFunctions(game)
        self.prob = cp.Problem(cp.Maximize(cp.sum(self.X[0] @ cp.bmat(self.objectifFunc))), self.constraints)

    def updateProb(self):
        self.prob = cp.Problem(cp.Maximize(cp.sum(self.X[0] @ cp.bmat(self.objectifFunc))), self.constraints)

    def init_variables(self):
        matrix = self.projectorConstraints()
        variablesDict = {}
        variable = [[None for i in range(self.n)] for j in range(self.n)]

        for line in range(self.n):
            for column in range(self.n):

                varId = matrix[line][column]
                if varId not in variablesDict:
                    variablesDict[varId] = cp.Variable()

                variable[line][column] = variablesDict[varId]

        return variable

    def projectorConstraints(self):
        '''
        Creation of projection constraints
        '''
        matrix = np.zeros((self.n, self.n))
        variableId = 0

        for i, Si in enumerate(self.S):
            for j, Sj in enumerate(self.S):
                var = Variable(self.S, i, j, self.game.operatorsPlayers)

                if var not in self.variableDict:
                    self.variableDict[var] = variableId
                    self.variablePosition[variableId] = (i, j)
                    variableId += 1

                matrix[i][j] = self.variableDict[var]

        return matrix


    def nashEquilibriumConstraint(self):
        '''
        Creation of Nash Equilibrium constraint
        '''
        for playerId in range(self.game.nbPlayers):

            payoutVec = []  # Payout if he follow advice
            for question in self.game.questions():
                for validAnswer in self.game.validAnswerIt(question):
                    payoutVec.append(self.game.genVecPlayerPayout(validAnswer, question, playerId))

            payoutVec = self.game.questionDistribution * np.array(payoutVec).transpose()

            # Payout for strat which diverge from advice
            for type in ['0', '1']:
                for noti in ['0', '1']:
                    payoutVecNot = []
                    for question in self.game.questions():
                        stillValid = lambda answer: question[playerId] != type or answer[playerId] != noti
                        nowValid = lambda answer: question[playerId] == type and answer[playerId] == noti

                        for validAnswer in filter(stillValid, self.game.validAnswerIt(question)):
                            payoutVecNot.append(self.game.genVecPlayerPayout(validAnswer, question, playerId))

                        for validAnswer in filter(nowValid, self.game.wrongAnswerIt(question)):
                            payoutVecNot.append(self.game.genVecPlayerNotPayout(validAnswer, question, playerId))

                    payoutVecNot = self.game.questionDistribution * np.array(payoutVecNot).transpose()
                    self.constraints.append(cp.sum(self.X[0] @ cp.bmat((payoutVec - payoutVecNot))) >= 0)

        self.updateProb()

    def objectifFunctions(self, game):
        objectifFunctionPayout = []

        for question in game.questions():
            for validAnswer in game.validAnswerIt(question):
                objectifFunctionPayout.append(game.genVecPayout(validAnswer, question))

        objectifFunction = self.game.questionDistribution * np.array(objectifFunctionPayout).transpose()

        return objectifFunction

    def optimize(self, verbose, warmStart):
        self.prob.solve(solver=cp.SCS, verbose=verbose, warm_start=warmStart)
        return self.prob.value
