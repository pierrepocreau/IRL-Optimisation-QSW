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
        self.X = cp.Variable((self.n, self.n), PSD=True) #Specifiying the constraint here divided by 2 the number of variables
        #self.constraints += [self.X == self.X.T] #Symmetric constraint already created by PSD
        #self.constraints += [self.X >> 0] #SDP
        self.constraints += [self.X[0][0] == 1] #Normalization

        self.objectifFunc = self.objectifFunctions(game)

    def projectorConstraints(self):
        '''
        Creation of projection constraints
        '''
        matrix = np.zeros((self.n, self.n))
        variableId = 0

        for i, Si in enumerate(self.S):
            for j, Sj in enumerate(self.S):

                if (i <= j): #Only half the matrix is important
                    var = Variable(self.S, i, j, self.game.operatorsPlayers)

                    if var not in self.variableDict:

                        self.variableDict[var] = variableId
                        self.variablePosition[variableId] = (i, j)
                        variableId += 1

                    matrix[i][j] = self.variableDict[var]

                    line, column = self.variablePosition[self.variableDict[var]]
                    if (i, j) != (line, column):
                        self.constraints += [self.X[i][j] == self.X[line][column]]

        return matrix #CVXPY can't create a set of variable from a matrix, so this return is useless


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
                    self.constraints.append(cp.sum(self.X[0] @ (payoutVec - payoutVecNot)) >= 0)

    def objectifFunctions(self, game):
        objectifFunctionPayout = []

        for question in game.questions():
            for validAnswer in game.validAnswerIt(question):
                objectifFunctionPayout.append(game.genVecPayout(validAnswer, question))

        objectifFunction = self.game.questionDistribution * np.array(objectifFunctionPayout).transpose()

        return objectifFunction

    def optimize(self, verbose):
        prob = cp.Problem(cp.Maximize(cp.sum(self.X[0] @ self.objectifFunc)), self.constraints)
        prob.solve(verbose=verbose)
        return prob.value
