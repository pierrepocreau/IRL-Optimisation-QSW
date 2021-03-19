import itertools
import numpy as np

class Game:

    def __init__(self, nbPlayers, v0, v1, sym=False):
        assert(nbPlayers == 3 or nbPlayers == 5)
        self.nbPlayers = nbPlayers
        self.questionDistribution = 1/(self.nbPlayers + 1)

        self.v0 = v0
        self.v1 = v1

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