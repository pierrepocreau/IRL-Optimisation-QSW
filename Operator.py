from itertools import groupby

def opToPlayer(op, operatorsPlayers):
    """
    Return the player Id with which the operator is associated.
    """
    assert(op != 0) #Id operator associated to every player.
    for i, operators in enumerate(operatorsPlayers):
        if op in operators:
            return i

class Variable:

    def __init__(self, S, i, j, playersOperators):
        """
        create the proba/variable <phi| Si^Dagger Sj |phi>
        """

        #We swap i and j if needed because the matrix is suppose to be symetric.
        if i <= j:
         self.Si = list(reversed(S[i]))
         self.Sj = S[j]
         self.value = self.Si + self.Sj
        else:
         self.Si = list(reversed(S[j]))
         self.Sj = S[i]
         self.value = self.Si + self.Sj

        self.playersOperators = playersOperators #Operators associated to each player
        self.cannonic = self.cannonicForm()

    def cannonicForm(self):
        cannonic = []

        operators = filter(lambda op: op!=0, self.value) #filter out identity
        operators = sorted(operators, key=lambda op: opToPlayer(op, self.playersOperators)) #Sort without commuting operators of a same player

        for k, g in groupby(operators):
            #GroupBy because the operators are projectors, so op^2 = op
            #if k so that 0 (which correspond to the Id operator) are ignore we put them at the end.
            if k: cannonic.append(k)

        #Fill the end with 0 (Id operators)
        while len(cannonic) != len(self.value):
            cannonic.append(0)

        return cannonic

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False

        #Two operator are equal if they have the same cannonic form.
        return other.cannonic == self.cannonic

    def __hash__(self):
        return tuple(self.cannonic).__hash__()