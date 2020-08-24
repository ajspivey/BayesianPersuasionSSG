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
import random

randThing = random.Random()
# randThing.seed(10)

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def generateRandomDefenders(defenderNum, targetNum, rewardCeiling=20, penaltyCeiling=20, costCeiling=10):
    defenders = list(range(1, defenderNum + 1))
    dRewards = {}
    dPenalties = {}
    dCosts = {}
    for m in defenders:
        dRewards[m] = []
        dPenalties[m] = []
        dCosts[m] = []
        for i in range(targetNum):
            dRewards[m].append(0)
            dPenalties[m].append(-1 * randThing.randint(1,penaltyCeiling))
            dCosts[m].append(-1 * randThing.randint(1,costCeiling))
    return defenders, dRewards, dPenalties, dCosts

def generateRandomAttackers(attackerNum, targetNum, rewardCeiling=20, penaltyCeiling=20):
    probability = 1.0
    attackers = list(range(1, attackerNum + 1))
    aRewards = {}
    aPenalties = {}
    q = []
    for a in attackers:
        if len(q) == attackerNum -1:
            q.append(probability)
        else:
            qVal = randThing.uniform(0,probability)
            probability -= qVal
            q.append(qVal)
        aRewards[a] = []
        aPenalties[a] = []
        for i in range(targetNum):
            aRewards[a].append(randThing.randint(1,rewardCeiling))
            aPenalties[a].append(-1 * randThing.randint(1,penaltyCeiling))
    return attackers, aRewards, aPenalties, q

def getPlacements():
    return list(permutations(defenders + ([-1] * (targetNum - len(defenders)))))

def getOmegaKeys():
    omegaKeys = []
    for aType in aTypes:
        for s in placements:
            for a in attackerActions:
                omegaKeys.append((s, a, aType))
    return omegaKeys

def defenderSocialUtility(s,k):
    costSum = 0
    for i in range(len(s)):
        defender = s[i]
        if defender != -1:
            costSum += dCosts[defender][i]
    utility = costSum # Defended
    if s[k] == -1: # Undefended
        utility = sum([penalty[k] for penalty in dPenalties.values()]) + costSum
    return utility

def utilityM(s,i,k,m):
    if s[k] == -1 and i != k: # Undefended
        utility = dPenalties[m][k] + dCosts[m][i]
    elif s[k] == m and i != k: # Left my post
        utility = dPenalties[m][k] + dCosts[m][i]
    else:
        utility = dCosts[m][i]
    return utility

start_time = getTime()
# ==============================================================================
# GAME SETTINGS
# ==============================================================================
targetNum = 3
defenderNum = 2
attackerNum = 2
M = 1000
defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(defenderNum, targetNum)
aTypes, aRewards, aPenalties, q = generateRandomAttackers(attackerNum, targetNum)

aTypes = [0,1]
aRewards = {0:[1,1,1], 1:[2,2,2]}
aPenalties = {0:[-1,-1,-1], 1:[-2,-2,-2]}
q = [0.7, 0.3]
# Defenders
defenders = [0,1]
dRewards = {0:[0,0,0],1:[0,0,0]}
dPenalties = {0:[-1,-1,-1],1:[-2,-2,-2]}
dCosts = {0:[-1,-1,-1],1:[-2,-2,-2]}

placements = getPlacements()
attackerActions = list(range(targetNum))
omegaKeys = getOmegaKeys()

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
# Create the model and objective function
model = Model('BayesianPersuasionSolver')
w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")

objectiveFunction = sum([q[lam - 1] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a) for s in placements for a in attackerActions]) for lam in aTypes])
# Add the constraints
# W constraints
model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])

# Solve the problem
model.maximize(objectiveFunction)
model.solve()
model.export("exampleModel.lp")
print("--- %s seconds ---" % (getTime() - start_time))
print(model.get_solve_status())
# for k, v in w.items():
#     print(k, float(v))
