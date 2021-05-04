from graph import readFile
import matplotlib.pylab as plt
import numpy as np


def plotHierarchie3():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    hierarchieNash = readFile('data/3Players_100Points_HierarchieNash.txt')
    x = np.linspace(0, 1, 100)

    graphState = list(map(lambda i: (1 + i)/2, x))

    plt.plot(x, hierarchieNash,'k-', label="Upper bound")
    plt.plot(x, graphState,'b-', label="SW graph state")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"\textit{social welfare}", labelpad=10)
    plt.ylim(0.4, 1)

    plt.title(r"Bound on the social welfare for $NC(C_3)$.")
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

    plt.plot(x, hierarchieNash,'k-', label="Upper bound")
    plt.plot(xGraphState, graphState,'b-', label="SW graph state")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"\textit{social welfare}", labelpad=10)
    plt.ylim(0.4, 1)

    plt.title(r"Bound on the social welfare for $NC_{00}(C_5)$.")
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

    plt.plot(x, hierarchieNash,'k-', label="Upper bound")
    plt.plot(xGraphState, graphState,'b-', label="SW graph state")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"\textit{social welfare}", labelpad=10)
    plt.ylim(0.4, 1)

    plt.title(r"Bound on the social welfare for $NC_{01}(C_5)$.")
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

    classicalfunc = lambda v0: ((8*v0 + 17) * (v0 <= 1/3) +
                           (6*v0 + 19) * (1/3 < v0 <= 1/2) +
                           (6*v0 + 19) * (1/2 < v0)) / 30

    classical = list(map(classicalfunc, x))


    xGraphState = list(filter(lambda i: i >= 1/2, x))
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))

    plt.plot(x, hierarchieNash,'k-', label="Upper bound")
    plt.plot(x, classical, linestyle="-", c="gray", label="Best Classical SW")
    plt.plot(xGraphState, graphState, c='blue', label="Graph state SW")
    plt.scatter(x, seesaw, marker='.', color = 'red', label="Seesaw")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"\textit{social welfare}", labelpad=10)
    plt.ylim(0.4, 1)

    plt.title(r"Bound and seesaw for $NC_{00}(C_5)$.")
    plt.grid()
    plt.savefig("5 players seesaw.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

def plotSeeSaw5Sym():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    seesaw = list(reversed(readFile("data/5Players_100Points_SymTrue_SeeSaw.txt")))
    hierarchieNash = readFile('data/5Players_100Points_SymTrue_HierarchieNash.txt')
    x = np.linspace(0, 1, 100)

    classicalfunc = lambda v0: 1/30*(4*v0 + 11)
    xclassical = list(filter(lambda i: i < 1/3, x))
    classical = list(map(classicalfunc, xclassical))

    xGraphState = list(filter(lambda i: i >= 1/3, x))
    graphState = list(map(lambda i: (1 + i)/2, xGraphState))

    plt.plot(x, hierarchieNash,'k-', label="Upper bound")
    plt.plot(xclassical, classical, linestyle="-", c="gray", label="Best classical SW")

    plt.plot(xGraphState, graphState,'b-', label="SW graph state")
    plt.scatter(x, seesaw, marker='.', color = 'r', label="seesaw")
    plt.legend(loc="best")

    plt.xlabel(r"$\frac{v_0}{v_1}$", labelpad=10)
    plt.ylabel(r"\textit{social welfare}", labelpad=10)
    plt.ylim(0.2, 1)

    plt.title(r"Bound and seesaw for $NC_{01}(C_5)$.")
    plt.grid()
    plt.savefig("5 players seesaw sym.png", dpi=300, pad_inches=.1, bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    plotHierarchie3()
    plotHierarchie5()
    plotHierarchie5Sym()
    plotSeeSaw5()
    plotSeeSaw5Sym()