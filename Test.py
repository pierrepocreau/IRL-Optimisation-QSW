from Operator import *
from itertools import product
import unittest
import numpy as np
from Game import Game
from SDP import SDP


class Test(unittest.TestCase):

    def testCanonicForm(self):
        operatorsP1 = [0, 1, 2]
        operatorsP2 = [0, 3, 4]
        operatorsP3 = [0, 5, 6]
        P3 = [operatorsP1, operatorsP2, operatorsP3]
        S = [list(s) for s in product(*P3)]

        proba = Variable(S, 3, 10, P3)
        proba3 = Variable(S, 8, 10, P3)

        #Test cannonic form and projection
        self.assertListEqual(proba.cannonic, [1, 3, 5, 0, 0, 0])
        self.assertListEqual(proba3.cannonic, [1, 4, 6, 5, 0, 0])

        #Test symetric
        self.assertEqual(Variable(S, 13, 24, P3), Variable(S, 24, 13, P3))
        self.assertNotEqual(Variable(S, 2, 10, P3), proba)

        #Test Id
        self.assertEqual(Variable(S, 10, 10, P3), Variable(S, 0, 10, P3))

    def testMatrixCreation(self):
        nbPlayers = 3
        v0, v1 = 1, 1
        operatorsP1 = [0, 1, 2]
        operatorsP2 = [0, 3, 4]
        operatorsP3 = [0, 5, 6]
        P3 = [operatorsP1, operatorsP2, operatorsP3]
        game = Game(nbPlayers, v0, v1, P3)
        sdp = SDP(game)
        matrix = sdp.projectorConstraints()

        self.assertListEqual(list(matrix[0,:]), list(range(27)))
        self.assertListEqual(list(matrix.diagonal()), list(range(27)))

    def testQuestions(self):
        nbPlayers = 3
        v0, v1 = 1, 1
        operatorsP1 = [0, 1, 2]
        operatorsP2 = [0, 3, 4]
        operatorsP3 = [0, 5, 6]
        P3 = [operatorsP1, operatorsP2, operatorsP3]
        game = Game(nbPlayers, v0, v1, P3)

        questions = list(game.questions())
        self.assertIn("111", questions)
        self.assertIn("100", questions)
        self.assertIn("010", questions)
        self.assertIn("001", questions)
        self.assertEqual(len(questions), 4)

    def testValidAnswer(self):
        nbPlayers = 3
        v0, v1 = 1, 1
        operatorsP1 = [0, 1, 2]
        operatorsP2 = [0, 3, 4]
        operatorsP3 = [0, 5, 6]
        P3 = [operatorsP1, operatorsP2, operatorsP3]
        game = Game(nbPlayers, v0, v1, P3)

        questions = list(game.questions())

        self.assertIn("111", questions)
        answer111 = list(game.validAnswerIt("111"))
        self.assertIn("100", answer111)
        self.assertIn("010", answer111)
        self.assertIn("001", answer111)
        self.assertIn("111", answer111)
        self.assertEqual(len(answer111), 4)

        self.assertIn("010", questions)
        answer010 = list(game.validAnswerIt("010"))
        self.assertIn("110", answer010)
        self.assertIn("011", answer010)
        self.assertIn("101", answer010)
        self.assertIn("000", answer010)
        self.assertEqual(len(answer010), 4)

    def testGenVec(self):
        nbPlayers = 3
        v0, v1 = 1, 1
        operatorsP1 = [0, 1, 2]
        operatorsP2 = [0, 3, 4]
        operatorsP3 = [0, 5, 6]
        P3 = [operatorsP1, operatorsP2, operatorsP3]
        game = Game(nbPlayers, v0, v1, P3)

        encodingVec = game.genVec("000", "111")
        correct = [0] * 26 + [1]
        self.assertListEqual(encodingVec, correct)

        encodingVec = game.genVec("000", "010")
        correct = [0] * 27
        correct[game.S.index([1, 4, 5])] = 1
        self.assertListEqual(encodingVec, correct)

        encodingVec = game.genVec("100", "010")
        correct = [0] * 27
        # P(100 | 010) = P(I00|010) - P(000|010)
        correct[game.S.index([0, 4, 5])] = 1
        correct[game.S.index([1, 4, 5])] = -1
        self.assertListEqual(encodingVec, correct)

        encodingVec = game.genVec("110", "010")
        correct = [0] * 27
        #P(110|010) = P(II0|010) - P(OOO|O1O) - P(010|010) - P(100|010) = ...
        correct[game.S.index([0, 0, 5])] = 1
        correct[game.S.index([1, 4, 5])] = -1
        correct[game.S.index([0, 4, 5])] = -1
        correct[game.S.index([1, 4, 5])] = 1
        correct[game.S.index([1, 0, 5])] = -1
        correct[game.S.index([1, 4, 5])] = +1

        self.assertListEqual(encodingVec, correct)

if __name__ == "__main__":
    unittest.main()
