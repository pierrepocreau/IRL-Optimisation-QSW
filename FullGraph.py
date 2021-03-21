import numpy as np
import matplotlib.pyplot as plt

def readFile(file):
    f = open(file, "r")
    data = []
    for element in f.read().split("\n"):
        try:
            data.append(float(element))
        except: pass
    return data

X = np.linspace(0, 1, 25)
GraphState = readFile("QSW_25Points_GraphState.txt")
SDPNash = readFile("QSW_25Points_SDP_WithNash.txt")

SDPNoNash = readFile("QSW_25Points_SDP_WithoutNash.txt")

SeeSaw = readFile("QSW_25Points_SeeSaw.txt")

print(len(X))
print(len(GraphState))
plt.plot(X, GraphState, label="GraphState")
plt.plot(X, SDPNash, label="SDPNash")
plt.plot(X, SDPNoNash, label="SDPNotNash")
plt.plot(X, SeeSaw, label="SeeSaw")
plt.xlabel("V0/V1")
plt.ylabel("QSW")
plt.legend(loc="upper right")

plt.show()