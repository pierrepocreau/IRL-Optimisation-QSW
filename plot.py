import quantumStrategies
from graph import readFile
import matplotlib.pylab as plt
import numpy as np
from cycler import cycler
from game import Game

def plotSeeSaw5():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    seesaw = list(reversed(readFile("data/5Players_100Points_SymFalse_SeeSaw.txt")))
    hierarchieNash = readFile('data/5Players_100Points_SymFalse_HierarchieNash.txt')
    x = np.linspace(0, 1, 100)
    abscisseFunc = lambda v0: v0 / (2 - v0)

    # classical strategies functions
    classicalfunc1 = lambda v0: (8*v0 + 17*(2 - v0)) /30
    classicalfunc2 = lambda v0: (6*v0 + 19*(2-v0)) /30

    # range where they are equlibriums
    xclassical1 = list(filter(lambda v0: v0/(2 - v0) < 1/3, x))
    xclassical2 = list(filter(lambda v0: v0/(2- v0) >= 1/3, x))

    # social welfare of those strategies
    classical1 = list(map(classicalfunc1, xclassical1))
    classical2 = list(map(classicalfunc2, xclassical2))

    # change abscisse
    xclassical1 = list(map(abscisseFunc, xclassical1))
    xclassical2 = list(map(abscisseFunc, xclassical2))

    xGraphState = list(filter(lambda v0: v0/(2 - v0) >= 1/2, x))
    graphState = list(map(lambda i: 1, xGraphState))
    xGraphState = list(map(abscisseFunc, xGraphState))

    x = list(map(abscisseFunc, x)) # x abscisse is v0/v1

    plt.scatter(x, seesaw, s=12, marker='x', color = 'k', linewidths=0.4, label="Seesaw lower bounds ($Q$)", zorder=3)

    plt.plot(x, hierarchieNash, "-", label="NPA upper bounds ($Q_{corr}$)")
    p1 = plt.plot(xclassical1, classical1, linestyle="-", label="Best classical SW")
    plt.plot(xclassical2, classical2, color = p1[0].get_color(), linestyle="-")

    plt.plot(xGraphState, graphState,'-', label="Graph state strat SW")
    plt.legend(loc="best")
    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"social welfare", labelpad=10)
    plt.ylim(0.8, 1.2)
    plt.xlim(0, 1)

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
    abscisseFunc = lambda v0: v0 / (2 - v0)

    classicalfunc = lambda v0: 1/12*(2*v0 + 7*(2 - v0))  #(Id 1 1)

    xclassical = list(x)
    classical = list(map(classicalfunc, xclassical))
    xclassical = list(map(abscisseFunc, xclassical))

    xGraphState = list(x)
    graphState = list(map(lambda v0: (2 - v0 + v0)/2, xGraphState))

    x = list(map(abscisseFunc, x))
    plt.scatter(x, seesaw, s=12, marker='x', color = 'k', linewidths=0.4, label="Seesaw lower bounds ($Q$)", zorder=3)
    plt.plot(x, hierarchieNash, "-", label="NPA upper bounds ($Q_{corr}$)")

    plt.plot(xclassical, classical, linestyle="-", label="Best classical SW")


    plt.plot(xGraphState, graphState,'-', label="Graph state strat SW")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"social welfare", labelpad=10)
    plt.ylim(0.7, 1.2)
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
    abscisseFunc = lambda v0: v0 / (2 - v0)

    classicalfunc1 = lambda v0: 1/30*(4*v0 + 11*(2 - v0))
    classicalfunc2 = lambda v0: 1/30*(5*v0 + 20*(2 - v0))


    xclassical1 = list(filter(lambda v0: v0/(2 - v0) < 1/3, x))
    xclassical2 = list(filter(lambda v0: v0/(2 - v0) >= 1/3, x))
    classical1 = list(map(classicalfunc1, xclassical1))
    classical2 = list(map(classicalfunc2, xclassical2))
    xclassical1 = list(map(abscisseFunc, xclassical1))
    xclassical2 = list(map(abscisseFunc, xclassical2))


    xGraphState = list(filter(lambda v0: v0 / (2 - v0) >= 1/3, x))
    graphState = list(map(lambda v0: 1, xGraphState))
    xGraphState = list(map(abscisseFunc, xGraphState))

    x = list(map(abscisseFunc, x))

    plt.scatter(x, seesaw, s=12, marker='x', color = 'k', linewidths=0.4, label="Seesaw lower bounds ($Q$)", zorder=3)

    plt.plot(x, hierarchieNash, "-", label="NPA upper bounds ($Q_{corr}$)")
    p1 = plt.plot(xclassical1, classical1, linestyle="-", label="Best classical SW")
    plt.plot(xclassical2, classical2, color = p1[0].get_color(), linestyle="-")


    plt.plot(xGraphState, graphState,'-', label="Graph state strat SW")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"social welfare", labelpad=10)
    plt.ylim(0.59, 1.2)
    plt.xlim(0, 1)


    plt.title(r"Upper and lower bounds for $NC_{01}(C_5)$.")
    plt.grid()
    plt.savefig("5 players sym.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    plotSeesaw3()
    plotSeeSaw5()
    plotSeeSaw5Sym()
