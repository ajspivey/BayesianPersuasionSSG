# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from docplex.mp.model import Model
from itertools import permutations
from functools import reduce
from operator import mul
from numpy import argmax
from time import time as getTime
import matplotlib.pyplot as plt
import random
import copy

# Internal Imports
from constants import DEFENDERNUM,ATTACKERNUM,TARGETNUM,AVGCOUNT,M,GAMEUTILITY,GAMEMODEL,GAMEEXTRAS
from util import generateRandomDefenders, generateRandomAttackers, numberToBase, \
                getPlacements, getOmegaKeys, defenderSocialUtility, utilityM, \
                aUtility, getLambdaPlacements, utilityDI, utilityLamI, \
                probabilityProtected, createGraph
from gameTypes import solveBPAllowOverlap, solveBPNoRequiredDefenderAssignment, \
                    solveBPNOOD, solveBPNOND, solveBaseline, solveBPNONDDualEllipsoid

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def iterateTargets(targetNum, avgCount, gameTypeList, overBudgets):
    utilities = [0] * len(gameTypeList)
    times = [0] * len(gameTypeList)
    for _ in range(avgCount):
        print(f"set {_} of avgCount")
        # Generate a new game
        defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, targetNum)
        aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, targetNum)

        gameIndex = 0
        for gameType, overBudget in zip(gameTypeList,overBudgets):
            if not overBudgets[gameIndex]:
                start = getTime()
                utilities[gameIndex] += gameType(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)[GAMEUTILITY]
                times[gameIndex] += getTime()-start
            else:
                utilities[gameIndex] = None
                times[gameIndex] = None
            gameIndex += 1
    for gameIndex in range(len(gameTypeList)):
        if not overBudgets[gameIndex]:
            utilities[gameIndex] /= avgCount
            times[gameIndex] /= avgCount

    return utilities,times

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
timeBudget = 10 # 30 minutes
utilities = []
minTargets = 2
maxTargets = 7

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
# Iterate over the targets
gameTypeList = [(solveBaseline,"Baseline", "y"), (solveBPAllowOverlap,"Allow Overlap", "r"), (solveBPNOND,"No Overlap, N Dummy Targets", "b"), (solveBPNONDDualEllipsoid,"Dual w/ Ellipsoid", "g")]
gameUtilities = {}
gameTimes = {}
for gameType in gameTypeList:
    gameUtilities[gameType[1]] = []
    gameTimes[gameType[1]] = []
overBudgets = [False] * len(gameTypeList)
for _ in range(minTargets, maxTargets):
    print(f"Iteration {minTargets} of {maxTargets} for targetCount")
    utilities, times = iterateTargets(targetNum=_, avgCount=AVGCOUNT, gameTypeList=[gameType[0] for gameType in gameTypeList], overBudgets=overBudgets)
    for gameIndex in range(len(overBudgets)):
        if times[gameIndex] is not None and times[gameIndex] > timeBudget:
            overBudgets[gameIndex] = True
        gameUtilities[gameTypeList[gameIndex][1]].append(utilities[gameIndex])
        gameTimes[gameTypeList[gameIndex][1]].append(times[gameIndex])
print(gameUtilities)
print(gameTimes)
uGraph = createGraph("Average Utilities", "Number of Targets", "Utility", gameTypeList, gameUtilities, xStart=2)
tGraph = createGraph("Average Runtimes", "Number of Targets", "Runtime", gameTypeList, gameTimes, xStart=2)
# Iterate over the defenders
# Iterate over the attackers

plt.show()
