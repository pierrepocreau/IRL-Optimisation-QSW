#Functions to generate the deviated strategy for 3 players

import numpy as np
from scipy.optimize import fmin

def QSW(v0, v1, theta, nbPlayers, sym=False):
    '''
    Calculations made in mathematica.
    '''
    assert(nbPlayers == 3 or nbPlayers == 5)
    if nbPlayers == 3:
        return 1 / 192 * (99 * v0 + 81 * v1 + 32 * (v0 - v1) * np.cos(theta) + 8 * (v0 - v1) * np.cos(
            2 * theta) + 5 * v0 * np.cos(4 * theta) + 7 * v1 * np.cos(4 * theta))

    if nbPlayers == 5:
        if not sym:
            return 1/6144 * (2874*v0 + 2278*v1 + 1984 * (v0 - v1) * np.cos(theta) -
                    2*(v0 + 249*v1)*np.cos(2 * theta) + 32*v0*np.cos(3 * theta) -
                    32*v1*np.cos(3 * theta) + 200*v0*np.cos(4 * theta) +
                    280*v1*np.cos(4 * theta) + 32*v0*np.cos(5 * theta) -
                    32*v1*np.cos(5 * theta) + 3*v0*np.cos(6 * theta) -
                    13*v1*np.cos(6 * theta) - 2*v0*np.cos(8 * theta) + 2*v1*np.cos(8 * theta) -
                    v0*np.cos(10 * theta) - v1*np.cos(10 * theta))
        else:
            return 1/6144 * (3178*v0 + 1974*v1 +
                    1504*(v0 - v1) * np.cos(theta) + (246*v0 - 746*v1) * np.cos(2 * theta) +
                    16*v0 * np.cos(3 * theta) - 16*v1 * np.cos(3 * theta) +
                    152*v0 * np.cos(4 * theta) + 328*v1 * np.cos(4 * theta) +
                    16*v0 * np.cos(5 * theta) - 16*v1 * np.cos(5 * theta) +
                    11*v0 * np.cos(6 * theta) - 21*v1 * np.cos(6 * theta) -
                    2*v0 * np.cos(8 * theta) + 2*v1 * np.cos(8 * theta) - v0 * np.cos(10 * theta) -
                    v1 * np.cos(10 * theta))

def DevNash(v0, v1, nbPlayers, sym=False):
    '''
    Value obtained via mathematica.
    I took them with my cursor on the grpah... But the equations are to complicated to copy from mathematica to python.
    Not very rigorous.
    '''
    assert(nbPlayers == 3 or nbPlayers == 5)
    if nbPlayers == 3:
        return v0/v1 >= 0.12

    if nbPlayers == 5:
        if sym:
            return v0/v1 >= 0.38

        else:
            return v0/v1 >= 0.507



def optimalTheta(v0, v1, nbPlayers, sym=False):
    '''
    Return the theta that maximize the deviated strategy for given v0 and v1.
    '''
    assert(nbPlayers == 3 or nbPlayers == 5)
    theta = fmin(lambda theta: - QSW(v0, v1, theta, nbPlayers, sym), np.pi/2, disp=False)
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


