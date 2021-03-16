import itertools
import numpy as np

class Game:

    def __init__(self, nbPlayers, v0, v1, operatorsPlayers, sym=False):
        assert(nbPlayers == 3 or nbPlayers == 5)
        self.nbPlayers = nbPlayers
        self.operatorsPlayers = operatorsPlayers
        self.S = [list(s) for s in itertools.product(*operatorsPlayers)]
        self.v0 = v0
        self.v1 = v1
        self.questionDistribution = 1/(self.nbPlayers + 1)

        self.sym = sym

    def questions(self):
        if self.sym:
            return self.questionsSym()
        else:
            return self.questionsClassic()

    def questionsClassic(self):
        """
        Generator on all question.
        """
        for i in range(self.nbPlayers):
            yield '0' * i + '1' + '0' * (self.nbPlayers - 1 - i)

        yield '1' * self.nbPlayers

    def questionsSym(self):
        for i in range(self.nbPlayers):
            secondOne = (i + 2) % self.nbPlayers
            i, secondOne = min(i, secondOne), max(i, secondOne)
            yield '0' * i + '1' + '0' * (secondOne - i - 1) + '1' + '0' * (self.nbPlayers - secondOne - 1)

        yield '1' * self.nbPlayers

    def validAnswer(self, answer, question):
        #When every player receives type1, an answer is correct when the sum of their answer is odd.
        if not '0' in question:
            return sum([int(bit) for bit in answer]) % 2

        #Otherwise, only one plauer receives type 1. An answer is correct when the sum of his answer and those of its
        #neighbors is even.
        playedType1 = question.index('1')

        # Symmetric question
        if question.count("1") == 2:
            if question[playedType1 + 2] != "1":
                playedType1 += 3

        involvedPlayers = [(playedType1 - 1) % self.nbPlayers, playedType1, (playedType1 + 1) % self.nbPlayers]

        parity = sum([int(answer[idx]) for idx in involvedPlayers]) % 2
        return not parity

    def validAnswerIt(self, question):
        for answer in itertools.product(['0', '1'], repeat=self.nbPlayers):
            answer = "".join(answer)
            if self.validAnswer(answer, question):
                yield answer

    def wrongAnswerIt(self, question):
        for answer in itertools.product(['0', '1'], repeat=self.nbPlayers):
            answer = "".join(answer)
            if not self.validAnswer(answer, question):
                yield answer

    def answerPayout(self, answer):
        '''
        Return the mean payout of an answer.
        '''
        nbOfOne = sum((int(bit) for bit in answer))
        return 1/self.nbPlayers * (nbOfOne * self.v1 + (self.nbPlayers - nbOfOne) * self.v0)

    def playerPayout(self, answer, playerId):
        '''
        Return the payout of a specific player.
        '''
        playerAnswer = answer[playerId]
        return self.v1 * (playerAnswer == '1') + self.v0 * (playerAnswer == '0')

    def notPlayerPayout(self, answer, playerId):
        '''
        Return the payout of an answer for a given question if the player i answer: not(adivce)
        '''
        playerAnswer = answer[playerId]
        return self.v0 * (playerAnswer == '1') + self.v1 * (playerAnswer == '0')

    def genVec(self, answer, question):
        '''
        Generate the encoding vector to get the probability of the answer given the question
        '''
        assert(len(answer) == len(question) == self.nbPlayers)

        vec = [0] * len(self.S)

        operator = []
        for p in range(self.nbPlayers):
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
        coef = self.playerPayout(answer, playerdId)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))

    def genVecPlayerNotPayout(self, answer, question, playerdId):
        """
        Payout of a player, if he not is answer.
        """
        coef = self.notPlayerPayout(answer, playerdId)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))

    def genVecPayout(self, answer, question):
        """
        Mean payout of all player.
        """
        coef = self.answerPayout(answer)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))