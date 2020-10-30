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

# random.seed(7)
# random.seed(18)
# random.seed(20)

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
game = solveBPNONDDualEllipsoid
game2 = solveBPAllowOverlap

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, TARGETNUM)
aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, TARGETNUM)
print(f"Program 1 Starting!")
game1Start = getTime()
score1, model, extras = game(TARGETNUM, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
game1Time = getTime()-game1Start
# print(f"Program 2 Starting!")
# game2Start = getTime()
# score2, model, extras = game2(TARGETNUM, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
# game2Time = getTime() - game2Start

print(f"Game 1 time: {game1Time}, game 1 utility: {score1}")
# print(f"Game 2 time: {game2Time}, game 2 utility: {score2}")
