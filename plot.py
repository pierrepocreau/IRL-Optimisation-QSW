import quantumStrategies
from graph import readFile
import matplotlib.pylab as plt
import numpy as np
from cycler import cycler


def plotHierarchie3():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    hierarchieNash = readFile('data/3Players_100Points_SymFalse_HierarchieNash.txt')
    x = np.linspace(0, 1, 100)

    graphState = list(map(lambda i: (1 + i)/2, x))

    plt.plot(x, hierarchieNash,'k-', label="Upper bounds")
    plt.plot(x, graphState,'b-', label="SW graph state")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"\textit{social welfare}", labelpad=10)
    plt.ylim(0.4, 1)
    plt.xlim(0, 1)

    plt.title(r"Bounds on the social welfare for $NC(C_3)$.")
    plt.grid()
    plt.savefig("3 players hierarchie.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

def plotHierarchie5():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    hierarchieNash = readFile('data/5Players_100Points_SymFalse_HierarchieNash.txt')
    x = np.linspace(0, 1, 100)

    xGraphState = list(filter(lambda i: i >= 1/2, x))
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))

    plt.plot(x, hierarchieNash,'k-', label="Upper bounds")
    plt.plot(xGraphState, graphState,'b-', label="SW graph state")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"\textit{social welfare}", labelpad=10)
    plt.ylim(0.4, 1)

    plt.title(r"Bounds on the social welfare for $NC_{00}(C_5)$.")
    plt.grid()
    plt.savefig("5 players hierarchie.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

def plotHierarchie5Sym():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    hierarchieNash = readFile('data/5Players_100Points_SymTrue_HierarchieNash.txt')
    x = np.linspace(0, 1, 100)

    xGraphState = list(filter(lambda i: i >= 1/3, x))
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))

    plt.plot(x, hierarchieNash,'k-', label="Upper bounds")
    plt.plot(xGraphState, graphState,'b-', label="SW graph state")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"\textit{social welfare}", labelpad=10)
    plt.ylim(0.4, 1)

    plt.title(r"Bounds on the social welfare for $NC_{01}(C_5)$.")
    plt.grid()
    plt.savefig("5 players hierarchie sym.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

#---------------------

def plotSeeSaw5():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    seesaw = list(reversed(readFile("data/5Players_100Points_SymFalse_SeeSaw.txt")))
    hierarchieNash = readFile('data/5Players_100Points_SymFalse_HierarchieNash.txt')
    x = np.linspace(0, 1, 100)

    classicalfunc1 = lambda v0: (8*v0 + 17) /30

    classicalfunc2 = lambda v0: (6*v0 + 19) /30

    xclassical1 = list(filter(lambda v0: v0 < 1/3, x))
    xclassical2 = list(filter(lambda v0: v0 >= 1/3, x))

    classical1 = list(map(classicalfunc1, xclassical1))
    classical2 = list(map(classicalfunc2, xclassical2))

    xDev = []
    QSW_dev = []
    v1 = 1
    nbPlayers = 5
    for v0 in x:
        if quantumStrategies.DevNash(v0, v1, nbPlayers):
            dev = quantumStrategies.QSW(v0, v1, quantumStrategies.optimalTheta(v0, v1, nbPlayers), nbPlayers)
            xDev.append(v0)
            QSW_dev.append(dev)


    xGraphState = list(filter(lambda i: i >= 1/2, x))
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))


    plt.scatter(x, seesaw, s=12, marker='x', color = 'k', linewidths=0.4, label="Seesaw lower bounds", zorder=3)

    plt.plot(x, hierarchieNash, "-", label="NPA upper bounds")
    p1 = plt.plot(xclassical1, classical1, linestyle="-", label="Best classical SW")
    plt.plot(xclassical2, classical2, color = p1[0].get_color(), linestyle="-")

    plt.plot(xDev, QSW_dev, "-", label="Deviated strat SW")

    plt.plot(xGraphState, graphState,'-', label="Graph state strat SW")
    plt.legend(loc="best")
    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"social welfare", labelpad=10)
    plt.ylim(0.55, 1)

    plt.title(r"Upper and lower bounds $NC_{00}(C_5)$.")
    plt.grid()
    plt.savefig("5 players.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

def plotSeesaw3():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    default_cycler = (cycler(color=['#0077BB', '#DDAA33', '#BB5566', '#000000']))
    plt.rc('axes', prop_cycle=default_cycler)

    seesaw = list(reversed(readFile("data/3Players_100Points_SymFalse_SeeSaw.txt")))
    hierarchieNash = readFile('data/3Players_100Points_SymFalse_HierarchieNash.txt')
    x = np.linspace(0, 1, 100)

    classicalfunc = lambda v0: 1/12*(2*v0 + 7)  #(Id 1 1)

    xclassical = list(x)
    classical = list(map(classicalfunc, xclassical))

    xGraphState = list(x)
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))

    xDev = []
    QSW_dev = []
    v1 = 1
    nbPlayers = 3
    for v0 in x:
        if quantumStrategies.DevNash(v0, v1, nbPlayers):
            dev = quantumStrategies.QSW(v0, v1, quantumStrategies.optimalTheta(v0, v1, nbPlayers), nbPlayers)
            xDev.append(v0)
            QSW_dev.append(dev)

    plt.scatter(x, seesaw, s=12, marker='x', color = 'k', linewidths=0.4, label="Seesaw lower bounds", zorder=3)
    plt.plot(x, hierarchieNash, "-", label="NPA upper bounds")

    plt.plot(xclassical, classical, linestyle="-", label="Best classical SW")

    plt.plot(xDev, QSW_dev, "-", label="Deviated strat SW")



    plt.plot(xGraphState, graphState,'-', label="Graph state strat SW")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"social welfare", labelpad=10)
    plt.ylim(0.4, 1)
    plt.xlim(0, 1)


    plt.title(r"Upper and lower bounds for $NC(C_3)$.")
    plt.grid()
    plt.savefig("3 players.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

def plotSeeSaw5Sym():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    seesaw = list(reversed(readFile("data/5Players_100Points_SymTrue_SeeSaw.txt")))
    hierarchieNash = readFile('data/5Players_100Points_SymTrue_HierarchieNash.txt')
    x = np.linspace(0, 1, 100)

    classicalfunc1 = lambda v0: 1/30*(4*v0 + 11)
    classicalfunc2 = lambda v0: 1/30*(5*v0 + 20)


    xclassical1 = list(filter(lambda i: i < 1/3, x))
    xclassical2 = list(filter(lambda i: i >= 1/3, x))
    classical1 = list(map(classicalfunc1, xclassical1))
    classical2 = list(map(classicalfunc2, xclassical2))

    xGraphState = list(filter(lambda i: i >= 1/3, x))
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))

    xDev = []
    QSW_dev = []
    v1 = 1
    nbPlayers = 5
    for v0 in x:
        if quantumStrategies.DevNash(v0, v1, nbPlayers, sym=True):
            dev = quantumStrategies.QSW(v0, v1, quantumStrategies.optimalTheta(v0, v1, nbPlayers, sym=True), nbPlayers, sym=True)
            xDev.append(v0)
            QSW_dev.append(dev)


    plt.scatter(x, seesaw, s=12, marker='x', color = 'k', linewidths=0.4, label="Seesaw lower bounds", zorder=3)

    plt.plot(x, hierarchieNash, "-", label="NPA upper bounds")
    p1 = plt.plot(xclassical1, classical1, linestyle="-", label="Best classical SW")
    plt.plot(xclassical2, classical2, color = p1[0].get_color(), linestyle="-")


    plt.plot(xDev, QSW_dev, "-", label="Deviated strat SW")

    plt.plot(xGraphState, graphState,'-', label="Graph state strat SW")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"social welfare", labelpad=10)
    plt.ylim(0.35, 1)

    plt.title(r"Upper and lower bounds for $NC_{01}(C_5)$.")
    plt.grid()
    plt.savefig("5 players sym.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()


#-----------------------------------------------------
def plotdev3():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    default_cycler = (cycler(color=['#0077BB', '#DDAA33', '#BB5566', '#000000']))
    plt.rc('axes', prop_cycle=default_cycler)

    x = np.linspace(0, 1, 100)

    classicalfunc = lambda v0: 1/12*(2*v0 + 7)  #(Id 1 1)
    xclassical = list(x)
    classical = list(map(classicalfunc, xclassical))

    xGraphState = list(x)
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))

    xDev = []
    QSW_dev = []
    v1 = 1
    nbPlayers = 3
    for v0 in x:
        if quantumStrategies.DevNash(v0, v1, nbPlayers):
            dev = quantumStrategies.QSW(v0, v1, quantumStrategies.optimalTheta(v0, v1, nbPlayers), nbPlayers)
            xDev.append(v0)
            QSW_dev.append(dev)

    plt.plot(xclassical, classical, linestyle="-", label="Best classical SW")
    plt.plot(xDev, QSW_dev, "-", label="Deviated strat SW")
    plt.plot(xGraphState, graphState,'-', label="Graph state strat SW")

    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"social welfare", labelpad=10)
    plt.ylim(0.5, 1)
    plt.xlim(0, 1)


    plt.title(r"Different strategies for $NC(C_3)$.")
    plt.grid()
    plt.savefig("3playersStrats.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

def plot5SymDev():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    x = np.linspace(0, 1, 1000)

    classicalfunc1 = lambda v0: 1/30*(4*v0 + 11)
    classicalfunc2 = lambda v0: 1/30*(5*v0 + 20)


    xclassical1 = list(filter(lambda i: i < 1/3, x))
    xclassical2 = list(filter(lambda i: i >= 1/3, x))
    classical1 = list(map(classicalfunc1, xclassical1))
    classical2 = list(map(classicalfunc2, xclassical2))

    xGraphState = list(filter(lambda i: i >= 1/3, x))
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))

    xDev = []
    QSW_dev = []
    v1 = 1
    nbPlayers = 5
    for v0 in x:
        if quantumStrategies.DevNash(v0, v1, nbPlayers, sym=True):
            dev = quantumStrategies.QSW(v0, v1, quantumStrategies.optimalTheta(v0, v1, nbPlayers, sym=True), nbPlayers, sym=True)
            xDev.append(v0)
            QSW_dev.append(dev)



    p1 = plt.plot(xclassical1, classical1, linestyle="-", label="Best classical SW")
    plt.plot(xclassical2, classical2, color = p1[0].get_color(), linestyle="-")


    plt.plot(xDev, QSW_dev, "-", label="Deviated strat SW")

    plt.plot(xGraphState, graphState,'-', label="Graph state strat SW")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"social welfare", labelpad=10)
    plt.ylim(0.35, 1)

    plt.xticks([i for i in np.linspace(0, 1, 11)] + [1/3])
    plt.xticks(rotation=45)
    plt.xlim(0, 1)

    plt.title(r"Different strategies for $NC_{01}(C_5)$.")
    plt.grid()
    plt.savefig("5playersSymStrats.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

def plot5Dev():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    x = np.linspace(0, 1, 1000)

    classicalfunc1 = lambda v0: (8*v0 + 17) /30
    classicalfunc2 = lambda v0: (6*v0 + 19) /30

    xclassical1 = list(filter(lambda v0: v0 < 1/3, x))
    xclassical2 = list(filter(lambda v0: v0 >= 1/3, x))

    classical1 = list(map(classicalfunc1, xclassical1))
    classical2 = list(map(classicalfunc2, xclassical2))

    xDev = []
    QSW_dev = []
    v1 = 1
    nbPlayers = 5
    for v0 in x:
        if quantumStrategies.DevNash(v0, v1, nbPlayers):
            dev = quantumStrategies.QSW(v0, v1, quantumStrategies.optimalTheta(v0, v1, nbPlayers), nbPlayers)
            xDev.append(v0)
            QSW_dev.append(dev)


    xGraphState = list(filter(lambda i: i >= 1/2, x))
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))

    p1 = plt.plot(xclassical1, classical1, linestyle="-", label="Best classical SW")
    plt.plot(xclassical2, classical2, color = p1[0].get_color(), linestyle="-")
    plt.plot(xDev, QSW_dev, "-", label="Deviated strat SW")
    plt.plot(xGraphState, graphState,'-', label="Graph state strat SW")

    plt.legend(loc="best")
    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"social welfare", labelpad=10)
    plt.ylim(0.55, 1)

    plt.xticks([i for i in np.linspace(0, 1, 11)] + [1/3])
    plt.xticks(rotation=45)
    plt.xlim(0, 1)

    plt.title(r"Different strategies for $NC_{00}(C_5)$.")
    plt.grid()
    plt.savefig("5PlayersStrats.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    #plotHierarchie3()
    plotSeesaw3()
    plotdev3()
    #plotHierarchie5()
    #plotHierarchie5Sym()
    plotSeeSaw5()
    plot5Dev()
    plotSeeSaw5Sym()
    plot5SymDev()