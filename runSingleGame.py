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

from gameTypes import solveBaseline, solvePrimalOverlap, solvePrimalNoOverlap, solveDualEllipsoid
# This has a much higher baseline for some reason
# random.seed(1)
# random.seed(7)
# random.seed(18)
# random.seed(20)

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
games = [(solveBaseline, "Baseline"), (solvePrimalOverlap, "Primal with Overlap"), (solvePrimalNoOverlap, "Primal with no Overlap"), (solveDualEllipsoid, "Dual with Ellipsoid")]

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, TARGETNUM)
aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, TARGETNUM)

for game, name in games:
    start = getTime()
    score, model, extras = game(TARGETNUM, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
    time = getTime() - start
    print(f"time for {name}: {time}")
    print(f"Score for {name}: {score}")
