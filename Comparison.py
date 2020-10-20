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
import copy

# Internal Imports
from constants import DEFENDERNUM,ATTACKERNUM,TARGETNUM,AVGCOUNT,M
from util import generateRandomDefenders, generateRandomAttackers, numberToBase, \
                getPlacements, getOmegaKeys, defenderSocialUtility, utilityM, \
                aUtility, getLambdaPlacements, utilityDI, utilityLamI, \
                probabilityProtected, createGraph
from gameTypes import solveBPAllowOverlap, solveBPNoRequiredDefenderAssignment, \
                    solveBPNOOD, solveBPNOND, solveBaseline

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def iterateTargets(targetNum, avgCount, bpOverbudget, bpNOODOverbudget, bpNONDOverbudget, bpNRDAOverbudget, baselineOverbudget):
    bpUtility = 0
    bpNOODUtility = 0
    bpNONDUtility = 0
    bpNRDAUtility = 0
    baselineUtility = 0
    bpTime = 0
    bpNOODTime = 0
    bpNONDTime = 0
    bpNRDATime = 0
    baselineTime = 0
    for _ in range(avgCount):
        print(f"set {_} of avgCount")
        # Generate a new game
        defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, targetNum)
        aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, targetNum)

        # BP Allow Overlap
        if not bpOverbudget:
            bpStart = getTime()
            bpScore, bpModel = solveBPAllowOverlap(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
            bpUtility += bpScore
            bpTime += getTime() - bpStart
        else:
            bpUtility = None
            bpTime = None

        # BP No Overlap 1 dummy target (Not always feasible)
        if not bpNOODOverbudget:
            bpNOODStart = getTime()
            # bpNOODScore, bpNOODModel = solveBPNOOD(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
            bpNOODScore = 0
            bpNOODUtility += bpNOODScore
            bpNOODTime += getTime() - bpNOODStart
        else:
            bpNOODUtility = None
            bpNOODTime = None

        # BP No Overlap n dummy targets (matches defender count)
        if not bpNONDOverbudget:
            bpNONDStart = getTime()
            bpNONDScore, bpNONDModel = solveBPNOND(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
            bpNONDUtility += bpNONDScore
            bpNONDTime += getTime() - bpNONDStart
        else:
            bpNONDUtility = None
            bpNONDTime = None

        # BP No Required Assignment
        if not bpNRDAOverbudget:
            bpNRDAStart = getTime()
            bpNRDAScore, bpNRDAModel = solveBPNoRequiredDefenderAssignment(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
            bpNRDAUtility += bpNRDAScore
            bpNRDATime += getTime() - bpNRDAStart
        else:
            bpNRDAUtility = None
            bpNRDATime = None

        # # Build the Baseline Model
        if not baselineOverbudget:
            baselineStart = getTime()
            baselineScore, baselineModels = solveBaseline(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
            baselineUtility += baselineScore
            baselineTime += getTime() - baselineStart
        else:
            baselineUtility = None
            baselineTime = None
    if not bpOverbudget:
        bpUtility /= avgCount
        bpTime /= avgCount
    if not bpNOODOverbudget:
        bpNOODUtility /= avgCount
        bpNOODTime /= avgCount
    if not bpNONDOverbudget:
        bpNONDUtility /= avgCount
        bpNONDTime /= avgCount
    if not bpNRDAOverbudget:
        bpNRDAUtility /= avgCount
        bpNRDATime /= avgCount
    if not baselineOverbudget:
        baselineUtility /= avgCount
        baselineTime /= avgCount
    return bpUtility, bpTime, bpNOODUtility, bpNOODTime, bpNONDUtility, bpNONDTime, bpNRDAUtility, bpNRDATime, baselineUtility, baselineTime

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
timeBudget = 10 # 30 minutes
defenderUtilities = []
solutionTimes = []
models = {}

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
avgBPUtils = []
avgBPTimes = []
avgbpNOODUtils = []
avgbpNOODTimes = []
avgbpNONDUtils = []
avgbpNONDTimes = []
avgBPNRDAUtils = []
avgBPNRDATimes = []
avgBaselineUtils = []
avgBaselineTimes = []
# Iterate over the targets
bpOver = False
bpNOODOver = False
bpNONDOver = False
bpNRDAOver = False
baselineOver = False
for _ in range(2, 9):
    print(f"Iteration {_} of {9} for targetCount")
    bpUtility, bpTime, bpNOODUtility, bpNOODTime, bpNONDUtility, bpNONDTime, bpNRDAUtility, bpNRDATime, baselineUtility, baselineTime = iterateTargets(targetNum=_, avgCount=AVGCOUNT, bpOverbudget=bpOver, bpNOODOverbudget=bpNOODOver, bpNONDOverbudget=bpNONDOver, bpNRDAOverbudget=bpNRDAOver, baselineOverbudget=baselineOver)
    if bpTime is not None and bpTime > timeBudget:
        bpOver = True
    if bpNOODTime is not None and bpNOODTime > timeBudget:
        bpNOODOver = True
    if bpNONDTime is not None and bpNONDTime > timeBudget:
        bpNONDOver = True
    if bpNRDATime is not None and bpNRDATime > timeBudget:
        bpNRDAOver = True
    if bpOver and bpNRDAOver:
        baselineOver = True
    avgBPUtils.append(bpUtility)
    avgBPTimes.append(bpTime)
    avgbpNOODUtils.append(bpNOODUtility)
    avgbpNOODTimes.append(bpNOODTime)
    avgbpNONDUtils.append(bpNONDUtility)
    avgbpNONDTimes.append(bpNONDTime)
    avgBPNRDAUtils.append(bpNRDAUtility)
    avgBPNRDATimes.append(bpNRDATime)
    avgBaselineUtils.append(baselineUtility)
    avgBaselineTimes.append(baselineTime)
    print(f"BP: {bpUtility}")
    print(f"NOOD: {bpNOODUtility}")
    print(f"NOND: {bpNONDUtility}")
    print(f"NRDA: {bpNRDAUtility}")
uGraph = createGraph("Average Utilities", "Number of Targets", "Utility", avgBPUtils, "BP Utility", avgbpNOODUtils, "BPRNOOD Utility", avgbpNONDUtils, "BPRNOND Utility", avgBPNRDAUtils, "BPNRD Utility", avgBaselineUtils, "Baseline Utility", xStart = 2)
tGraph = createGraph("Average Runtimes", "Number of Targets", "Runtime", avgBPTimes, "Persuasion Scheme Time", avgbpNOODTimes, "BPRNOOD Time", avgbpNONDTimes, "BPRNOND Time", avgBPNRDATimes, "BPNRD Time", avgBaselineTimes, "Baseline Time", xStart = 2)
# Iterate over the defenders
# Iterate over the attackers

plt.show()
