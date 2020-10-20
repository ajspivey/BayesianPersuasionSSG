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
from constants import DEFENDERNUM,ATTACKERNUM,TARGETNUM,AVGCOUNT,M
from util import generateRandomDefenders, generateRandomAttackers, numberToBase, \
                getPlacements, getOmegaKeys, defenderSocialUtility, utilityM, \
                aUtility, getLambdaPlacements, utilityDI, utilityLamI, \
                probabilityProtected, createGraph
from gameTypes import solveBPAllowOverlap, solveBPNoRequiredDefenderAssignment, \
                    solveBPNOOD, solveBPNOND, solveBaseline, solveBPNONDDualEllipsoid

random.seed(0)

# ==============================================================================
# FUNCTIONS
# ==============================================================================
defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, TARGETNUM)
aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, TARGETNUM)
score, model = solveBPNONDDualEllipsoid(TARGETNUM, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
