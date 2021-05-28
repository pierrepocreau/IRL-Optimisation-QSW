def bestClassicalStrategy(game):
    # Social welfare of the best classical strategy for a given game.
    v0, v1 = game.v0.value, game.v1.value
    payoutRatio = v0 / v1

    assert(game.nbPlayers == 3 or game.nbPlayers == 5)
    assert(0 <= payoutRatio <= 1)

    if game.nbPlayers == 5:
        if not game.sym:
            if payoutRatio <= 1 / 3:
                socialWelfare = 1/30 * (8 * v0 + 17 * v1)
            else:
                socialWelfare = 1/30 * (6 * v0 + 19 * v1)
        else:
            if payoutRatio <= 1 / 3:
                socialWelfare = 1 / 30 * (4 * v0 + 11 * v1)
            else:
                socialWelfare = 1 / 30 * (5 * v0 + 20 * v1)
    else:
        socialWelfare = 1/12 * (7 * v1 + 2 * v0)

    return socialWelfare

