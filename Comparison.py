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
def iterateTargets(iterable, avgCount, gameTypeList, overBudgets):
    utilities = [0] * len(gameTypeList)
    times = [0] * len(gameTypeList)
    for _ in range(avgCount):
        print(f"set {_} of {avgCount} for {iterable}")
        # Generate a new game
        defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, iterable)
        aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, iterable)

        gameIndex = 0
        for gameType, overBudget in zip(gameTypeList,overBudgets):
            if not overBudgets[gameIndex]:
                start = getTime()
                utilities[gameIndex] += gameType(iterable, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)[GAMEUTILITY]
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

def iterateDefenders(iterable, avgCount, gameTypeList, overBudgets):
    utilities = [0] * len(gameTypeList)
    times = [0] * len(gameTypeList)
    for _ in range(avgCount):
        print(f"set {_} of {avgCount} for {iterable}")
        # Generate a new game
        defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(iterable, TARGETNUM)
        aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, TARGETNUM)

        gameIndex = 0
        for gameType, overBudget in zip(gameTypeList,overBudgets):
            if not overBudgets[gameIndex]:
                start = getTime()
                utilities[gameIndex] += gameType(TARGETNUM, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)[GAMEUTILITY]
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

def iterateAttackers(iterable, avgCount, gameTypeList, overBudgets):
    utilities = [0] * len(gameTypeList)
    times = [0] * len(gameTypeList)
    for _ in range(avgCount):
        print(f"set {_} of {avgCount} for {iterable}")
        # Generate a new game
        defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, TARGETNUM)
        aTypes, aRewards, aPenalties, q = generateRandomAttackers(iterable, TARGETNUM)

        gameIndex = 0
        for gameType, overBudget in zip(gameTypeList,overBudgets):
            if not overBudgets[gameIndex]:
                start = getTime()
                utilities[gameIndex] += gameType(TARGETNUM, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)[GAMEUTILITY]
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

def generateGraph(gameTypeList, iterationFunction, iterableLabel):
    gameUtilities = {}
    gameTimes = {}
    for gameType in gameTypeList:
        gameUtilities[gameType[1]] = []
        gameTimes[gameType[1]] = []
    overBudgets = [False] * len(gameTypeList)
    for _ in range(minIterable, maxIterable):
        print(f"Iteration {_} of {maxIterable} for targetCount")
        utilities, times = iterationFunction(iterable=_, avgCount=AVGCOUNT, gameTypeList=[gameType[0] for gameType in gameTypeList], overBudgets=overBudgets)
        for gameIndex in range(len(overBudgets)):
            if times[gameIndex] is not None and times[gameIndex] > timeBudget:
                overBudgets[gameIndex] = True
            gameUtilities[gameTypeList[gameIndex][1]].append(utilities[gameIndex])
            gameTimes[gameTypeList[gameIndex][1]].append(times[gameIndex])
    print(gameUtilities)
    print(gameTimes)
    uGraph = createGraph(f"Average Utilities ({iterableLabel})", f"Number of {iterableLabel}", "Utility", gameTypeList, gameUtilities, xStart=minIterable)
    tGraph = createGraph(f"Average Runtimes ({iterableLabel})", f"Number of {iterableLabel}", "Runtime", gameTypeList, gameTimes, xStart=minIterable)
    plt.show()


# ==============================================================================
# GAME SETTINGS
# ==============================================================================
timeBudget = 100 # 30 minutes
utilities = []
minIterable = 2
maxIterable = 6

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
gameTypeList = [(solveBaseline,"Baseline", "y"), (solveBPAllowOverlap,"Allow Overlap", "r"), (solveBPNONDDualEllipsoid,"Dual w/ Ellipsoid", "g")]
# Iterate over the targets
# generateGraph(gameTypeList, iterateTargets, "Targets")
generateGraph(gameTypeList, iterateDefenders, "Defenders")
# generateGraph(gameTypeList, iterateAttackers, "Attackers")
