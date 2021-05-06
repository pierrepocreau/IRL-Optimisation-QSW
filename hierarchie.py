import cvxpy as cp
import numpy as np
from canonicOp import CanonicMonome

import itertools

class Hierarchie:

    def __init__(self, game, operatorsPlayers):
        self.game = game

        self.operatorsPlayers = operatorsPlayers
        self.monomeList = [list(s) for s in itertools.product(*operatorsPlayers)]
        self.n = len(self.monomeList)

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

        for i, Si in enumerate(self.monomeList):
            for j, Sj in enumerate(self.monomeList):
                var = CanonicMonome(self.monomeList, i, j, self.operatorsPlayers)

                if var not in self.variableDict:
                    self.variableDict[var] = variableId
                    self.variablePosition[variableId] = (i, j)
                    variableId += 1

                matrix[i][j] = self.variableDict[var]

        return matrix


    def setNashEqConstraints(self):
        '''
        Creation of Nash Equilibrium constraint
        '''
        for playerId in range(self.game.nbPlayers):

            payoutVec = []  # Payout if he follow advice
            for question in self.game.questions():
                for validAnswer in self.game.validAnswerIt(question):
                    payoutVec.append(self.genVecPlayerPayoutWin(validAnswer, question, playerId))

            payoutVec = self.game.questionDistribution * np.array(payoutVec).transpose()

            # Payout for strat which diverge from advice
            for type in ['0', '1']:
                for noti in ['0', '1']:
                    payoutVecNot = []
                    for question in self.game.questions():
                        #Answers where the player doesn't not his answer
                        untouchedAnswers = lambda answer: question[playerId] != type or answer[playerId] != noti

                        #Answers where the player not his answer
                        notAnswers = lambda answer: question[playerId] == type and answer[playerId] == noti

                        # if he is not involved, the set of accepted answer is the same
                        if playerId not in self.game.involvedPlayers(question):
                            #The player is not involved, the set of accepted question stay the same
                            for validAnswer in filter(untouchedAnswers, self.game.validAnswerIt(question)):
                                payoutVecNot.append(self.genVecPlayerPayoutWin(validAnswer, question, playerId))

                            for validAnswer in filter(notAnswers, self.game.validAnswerIt(question)):
                                payoutVecNot.append(self.genVecPlayerNotPayoutWin(validAnswer, question, playerId))

                        #The player is involved. He loose on the accepted answer where he nots. But some rejected answers
                        #are now accepted.
                        else:
                            for validAnswer in filter(untouchedAnswers, self.game.validAnswerIt(question)):
                                payoutVecNot.append(self.genVecPlayerPayoutWin(validAnswer, question, playerId))

                            for validAnswer in filter(notAnswers, self.game.wrongAnswerIt(question)):
                                payoutVecNot.append(self.genVecPlayerNotPayoutWin(validAnswer, question, playerId))

                    payoutVecNot = self.game.questionDistribution * np.array(payoutVecNot).transpose()
                    self.constraints.append(cp.sum(self.X[0] @ cp.bmat((payoutVec - payoutVecNot))) >= 0)

        self.updateProb()


    def genVec(self, answer, question):
        '''
        Generate the encoding vector to get the probability of the answer given the question
        '''
        assert(len(answer) == len(question) == self.game.nbPlayers)

        vec = [0] * len(self.monomeList)

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
            if operator in self.monomeList:
                vec[self.monomeList.index(operator)] = coef

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

    def genVecPlayerPayoutWin(self, answer, question, playerdId):
        """
        Return the vector with which to multiply the first row of X to have the payout of a player.
        """
        coef = self.game.playerPayoutWin(answer, playerdId)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))

    def genVecPlayerNotPayoutWin(self, answer, question, playerdId):
        """
        Payout of a player, if he not is answer.
        """
        coef = self.game.notPlayerPayoutWin(answer, playerdId)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))

    def genVecWelfareWin(self, answer, question):
        """
        Mean payout of all player.
        """
        coef = self.game.answerPayoutWin(answer)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))

    def objectifFunctions(self, game):
        objectifFunctionPayout = []

        for question in game.questions():
            for validAnswer in game.validAnswerIt(question):
                objectifFunctionPayout.append(self.genVecWelfareWin(validAnswer, question))

        objectifFunction = self.game.questionDistribution * np.array(objectifFunctionPayout).transpose()

        return objectifFunction

    def optimize(self, verbose, warmStart, solver):
        assert(solver == "SCS" or solver == "MOSEK")
        if solver == "SCS":
            self.prob.solve(solver=cp.SCS, verbose=verbose, warm_start=warmStart)
        else:
            self.prob.solve(solver=cp.MOSEK, verbose=verbose, warm_start=warmStart)
        return self.prob.value
