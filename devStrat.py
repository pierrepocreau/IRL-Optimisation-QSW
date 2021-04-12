#Functions to generate the deviated strategy for 3 players

import numpy as np
from mainSeeSaw import quantumEqCheck
from scipy.optimize import fmin

def QSW(v0, v1, theta, nbPlayers):
    assert(nbPlayers == 3)
    return 1 / 192 * (99 * v0 + 81 * v1 + 32 * (v0 - v1) * np.cos(theta) + 8 * (v0 - v1) * np.cos(
        2 * theta) + 5 * v0 * np.cos(4 * theta) + 7 * v1 * np.cos(4 * theta))

def optimalTheta(v0, v1, nbPlayers):
    '''
    Return the theta that maximize the deviated strategy for given v0 and v1.
    '''
    assert(nbPlayers == 3)
    theta = fmin(lambda theta: - QSW(v0, v1, theta, nbPlayers) , np.pi/2, disp=False)
    return theta

def generatePOVMs(theta, nbPlayers):
    assert(nbPlayers == 3)
    psi = np.array([np.cos(theta/2), np.sin(theta/2)])
    POVMs_Dict = {}
    for player in range(nbPlayers):
        POVMs_Dict[str(player) + '00'] = np.array([[1, 0], [0,0]])
        POVMs_Dict[str(player) + '10'] = np.array([[0, 0], [0,1]])
        POVMs_Dict[str(player) + '01'] = np.outer(psi, psi)
        POVMs_Dict[str(player) + '11'] = np.eye(2) - np.outer(psi, psi)
    return POVMs_Dict

def generateRho(theta, nbPlayers):
    assert(nbPlayers == 3)
    state = np.array([np.cos(theta/2)**3, np.cos(theta/2)**2 * np.sin(theta/2),
                              np.cos(theta/2)**2 * np.sin(theta/2), -np.cos(theta/2) * np.sin(theta/2)**2,
                              np.cos(theta/2)**2 * np.sin(theta/2),  -np.cos(theta/2) * np.sin(theta/2)**2,
                              -np.cos(theta / 2) * np.sin(theta / 2) ** 2, -np.sin(theta/2)**3])
    return np.outer(state, state)


def qEqTest(v0, v1, nbPlayers):
    '''
    return True if the quantum strat deviated from graphstate is a quantum equilibrium for given v0 and v1.
    '''
    theta = optimalTheta(v0, v1, nbPlayers)
    POVMs_Dict = generatePOVMs(theta, nbPlayers)
    rho = generateRho(theta, nbPlayers)
    return quantumEqCheck(nbPlayers, v0, v1, POVMs_Dict, rho, 1E-6)

if __name__ == '__main__':
    nbPlayers = 3
    v1 = 1
    v0s = np.linspace(0, 1, 100)
    for v0 in v0s:
        print("Qeq test for deviated strategy for v0 = {}: {}".format(v0, qEqTest(v0, v1, nbPlayers)))


