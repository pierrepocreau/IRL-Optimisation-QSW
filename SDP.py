import cvxpy as cp
import numpy as np
from Operator import Variable
import itertools

class SDP:

    def __init__(self, game, operatorsPlayers):
        self.game = game

        self.operatorsPlayers = operatorsPlayers
        self.S = [list(s) for s in itertools.product(*operatorsPlayers)]
        self.n = len(self.S)

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
                var = Variable(self.S, i, j, self.operatorsPlayers)

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
                    payoutVec.append(self.genVecPlayerPayout(validAnswer, question, playerId))

            payoutVec = self.game.questionDistribution * np.array(payoutVec).transpose()

            # Payout for strat which diverge from advice
            for type in ['0', '1']:
                for noti in ['0', '1']:
                    payoutVecNot = []
                    for question in self.game.questions():
                        stillValid = lambda answer: question[playerId] != type or answer[playerId] != noti
                        nowValid = lambda answer: question[playerId] == type and answer[playerId] == noti

                        for validAnswer in filter(stillValid, self.game.validAnswerIt(question)):
                            payoutVecNot.append(self.genVecPlayerPayout(validAnswer, question, playerId))

                        for validAnswer in filter(nowValid, self.game.wrongAnswerIt(question)):
                            payoutVecNot.append(self.genVecPlayerNotPayout(validAnswer, question, playerId))

                    payoutVecNot = self.game.questionDistribution * np.array(payoutVecNot).transpose()
                    self.constraints.append(cp.sum(self.X[0] @ cp.bmat((payoutVec - payoutVecNot))) >= 0)

        self.updateProb()


    def genVec(self, answer, question):
        '''
        Generate the encoding vector to get the probability of the answer given the question
        '''
        assert(len(answer) == len(question) == self.game.nbPlayers)

        vec = [0] * len(self.S)

        operator = []
        for p in range(self.game.nbPlayers):
            #the flag is negative if the player answer 1, positive otherwise.
            flag = -2 * (answer[p] == "1") + 1

            if question[p] == "1":
                operator.append(flag * (p + 1) * 2)
            else:
                operator.append(flag * (p * 2 + 1))

        def recursiveFunc(operator, coef):
            #The operator is in the matrix
            if operator in self.S:
                vec[self.S.index(operator)] = coef

            #There is a negative number as operator (the player answer 1)
            else:
                #We find the negative operator
                negIdx = next(idx for idx, x in enumerate(operator) if x < 0)

                opId = operator.copy()
                opId[negIdx] = 0
                op2 = operator.copy()
                op2[negIdx] = - op2[negIdx]

                #P(101|111) = P(I01|111) - P(OO1|111)
                recursiveFunc(opId, coef)
                recursiveFunc(op2, -coef)

        recursiveFunc(operator, 1)
        return vec

    def genVecPlayerPayout(self, answer, question, playerdId):
        """
        Return the vector with which to multiply the first row of X to have the payout of a player.
        """
        coef = self.game.playerPayout(answer, playerdId)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))

    def genVecPlayerNotPayout(self, answer, question, playerdId):
        """
        Payout of a player, if he not is answer.
        """
        coef = self.game.notPlayerPayout(answer, playerdId)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))

    def genVecPayout(self, answer, question):
        """
        Mean payout of all player.
        """
        coef = self.game.answerPayout(answer)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))

    def objectifFunctions(self, game):
        objectifFunctionPayout = []

        for question in game.questions():
            for validAnswer in game.validAnswerIt(question):
                objectifFunctionPayout.append(self.genVecPayout(validAnswer, question))

        objectifFunction = self.game.questionDistribution * np.array(objectifFunctionPayout).transpose()

        return objectifFunction

    def optimize(self, verbose, warmStart):
        self.prob.solve(solver=cp.SCS, verbose=verbose, warm_start=warmStart)
        return self.prob.value
