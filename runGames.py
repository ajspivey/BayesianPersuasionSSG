# ==============================================================================
# IMPORTS
# ==============================================================================
from time import time as getTime
import sys

from gameTypes import solveBaseline, solvePostOverlap, solvePostNoOverlap, \
                    solvePostDualEllipsoid, solveExOverlap, solveExNoOverlap, \
                    solveExDualEllipsoid, solveExCompact
from generateGames import getFileName, readGames

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
methodDict = {
    "baseline": solveBaseline,
    "postOverlap": solvePostOverlap,
    "postNoOverlap": solvePostNoOverlap,
    "postDual": solvePostDualEllipsoid,
    "exOverlap": solveExOverlap,
    "exNoOverlap": solveExNoOverlap,
    "exDual": solveExDualEllipsoid,
    "compact": solveExCompact
}

def main():
    # Get the arguments
    args = sys.argv[1:]
    # check if we start from the beginning or not
    gameNum = 0
    avgTime = 0
    avgUtilityPerDefender = 0
    avgUtilityPerAttacker = 0
    if len(args) == 4:
        methodName, tNum, aNum, dNum = args
    else:
        methodName, tNum, aNum, dNum, startIndex, avgTime, avgUtilityPerDefender, avgUtilityPerAttacker = args
        avgTime, avgUtilityPerDefender, avgUtilityPerAttacker = [float(arg) for arg in [avgTime, avgUtilityPerDefender, avgUtilityPerAttacker]]
        gameNum = int(startIndex)
    tNum, aNum, dNum = [int(arg) for arg in [tNum,aNum,dNum]]
    method = methodDict[methodName]

    # Games location:
    fileName = getFileName(tNum,aNum,dNum)

    # Read in the games
    numGames, games = readGames(fileName)

    print(numGames)

    # Run the games and time them
    while gameNum < numGames:
        print(f"Running game {gameNum} for method {methodName}")
        game = games[gameNum]
        defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q = game
        start = getTime()
        utilityPerDefender, utilityPerAttacker, extras = method(tNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
        time = getTime() - start
        avgTime += time
        avgUtilityPerDefender += utilityPerDefender
        avgUtilityPerAttacker += utilityPerAttacker
        print(f"current sums (not divided yet): time: {avgTime}, defUtil: {avgUtilityPerDefender}, attUtil: {avgUtilityPerAttacker}")
        gameNum += 1
    avgTime /= numGames
    avgUtilityPerDefender /= numGames
    avgUtilityPerAttacker /= numGames
    print(f"Average time for method {methodName} with {tNum} targets, {aNum} attackers, and {dNum} defenders: {avgTime}")
    print(f"Average utility per defender for method {methodName} with {tNum} targets, {aNum} attackers, and {dNum} defenders: {avgUtilityPerDefender}")
    print(f"Average utility per attacker for method {methodName} with {tNum} targets, {aNum} attackers, and {dNum} defenders: {avgUtilityPerAttacker}")

# =========
# MAIN CALL
# =========
if __name__ == "__main__":
    main()
