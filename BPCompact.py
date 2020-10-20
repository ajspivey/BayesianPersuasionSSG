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

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def generateRandomDefenders(defenderNum, targetNum, rewardCeiling=20, penaltyCeiling=20, costCeiling=10):
    defenders = list(range(defenderNum))
    dRewards = {}
    dPenalties = {}
    dCosts = {}
    for m in defenders:
        dRewards[m] = []
        dPenalties[m] = []
        dCosts[m] = []
        for i in range(targetNum):
            dRewards[m].append(0)
            dPenalties[m].append(-1 * random.randint(1,penaltyCeiling))
            dCosts[m].append(-1 * random.randint(1,costCeiling))
    return defenders, dRewards, dPenalties, dCosts

def generateRandomAttackers(attackerNum, targetNum, rewardCeiling=20, penaltyCeiling=20):
    probability = 1.0
    attackers = list(range(attackerNum))
    aRewards = {}
    aPenalties = {}
    q = []
    for a in attackers:
        if len(q) == attackerNum -1:
            q.append(probability)
        else:
            qVal = random.uniform(0,probability)
            probability -= qVal
            q.append(qVal)
        aRewards[a] = []
        aPenalties[a] = []
        for i in range(targetNum):
            aRewards[a].append(random.randint(1,rewardCeiling))
            aPenalties[a].append(-1 * random.randint(1,penaltyCeiling))
    return attackers, aRewards, aPenalties, q

def getOmegaKeys(defenders, targetNum, aTypes):
    return [(m,i,lam) for m in defenders for i in rage(targetNum) for lam in aTypes]

start_time = getTime()
# ==============================================================================
# GAME SETTINGS
# ==============================================================================
targetNum = 6
defenderNum = 4
attackerNum = 3
M = 1000
defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(defenderNum, targetNum)
aTypes, aRewards, aPenalties, q = generateRandomAttackers(attackerNum, targetNum)
omegaKeys = getOmegaKeys(defenders,targetNum, aTypes)

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
# Create the model and objective function
model = Model('BayesianPersuasionSolverCompact')
w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
model.add_constraints([] <= 1 for i in range(targetNum)], names=[f"{a}"])
# Solve the problem
model.maximize(objectiveFunction)
model.solve()
model.export("exampleModel.lp")
print("--- %s seconds ---" % (getTime() - start_time))
print(model.get_solve_status())
# for k, v in w.items():
#     print(k, float(v))
