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

def getPlacements():
    return list(permutations(defenders + ([-1 * x for x in range(targetNum - len(defenders))])))

def getLambdaPlacements():
    lambdaPlacements = []
    for aType in aTypes:
        for s in placements:
            lambdaPlacements.append((aType, s))
    return lambdaPlacements

def utilityDI(m,x,i):
    utility = x[i] * (dRewards[m][i] + dCosts[m][i]) + (1-x[i]) * (dPenalties[m][i] + dCosts[m][i])
    return utility

def utilityLamI(x,lam,i):
    utility = x[i] * aPenalties[lam][i] + (1-x[i]) * aRewards[lam][i]
    return utility

def probabilityProtected(dStrats, targetNum):
    protectionOdds = []
    strats = list(zip(*dStrats.values()))
    for probabilities in strats:
        protectionOdds.append(1-reduce(mul, [1-odd for odd in probabilities]))
    return protectionOdds

start_time = getTime()
# ==============================================================================
# GAME SETTINGS
# ==============================================================================
targetNum = 3
defenderNum = 2
attackerNum = 1
M = 9999
defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(defenderNum, targetNum)
aTypes, aRewards, aPenalties, q = generateRandomAttackers(attackerNum, targetNum)
placements = getPlacements()
lambdaPlacements = getLambdaPlacements()

# ==============================================================================
# Defender Strategies
# ==============================================================================
dStrat = {}
models = {}
for m in defenders:
    model = Model(f"defenderStrategy{m}")
    x = model.continuous_var_list(keys=targetNum, lb=0, ub=1, name=f"x{m}")
    h = model.binary_var_dict(keys=[(lam, k) for lam in aTypes for k in range(targetNum)], lb=0, ub=1, name=f"h{m}")
    ul = model.continuous_var_dict(keys=aTypes, lb=-model.infinity, name=f"ua{m}")
    ud = model.continuous_var_dict(keys=[lam for lam in aTypes], lb=-model.infinity, name=f"ud{m}")
    objectiveFunction = sum([q[lam] * ud[lam] for lam in aTypes])
    model.add_constraints([ud[lam] <= utilityDI(m,x,i) + (1-h[(lam,i)]) * M for i in range(targetNum) for lam in aTypes], names=[f"defender utility for lam {lam}, i {i}" for i in range(targetNum) for lam in aTypes])
    model.add_constraints([ul[lam] <= utilityLamI(x,lam,i) + (1-h[(lam,i)]) * M for i in range(targetNum) for lam in aTypes], names=[f"lam {lam} utility leq for i {i}" for i in range(targetNum) for lam in aTypes])
    model.add_constraints([ul[lam] >= utilityLamI(x,lam,i) for i in range(targetNum) for lam in aTypes], names=[f"lam {lam} utility geq, for i {i}" for i in range(targetNum) for lam in aTypes])
    model.add_constraints([sum([h[(lam,i)] for i in range(targetNum)]) == 1 for lam in aTypes], names=[f"h sum is 1 for lam {lam}" for lam in aTypes])
    model.add_constraint(sum([x[i] for i in range(targetNum)]) == 1)
    # Solve the problem
    model.maximize(objectiveFunction)
    model.solve()
    model.export(f"modelBaseline{m}.lp")
    dStrat[m] = list([float(xVal) for xVal in x])
    models[m] = model
# Attacker response
aStrat = 0
protectionOdds = probabilityProtected(dStrat, targetNum)
for lam in aTypes:
    expectedUtilities = []
    for i in range(targetNum):
        expectedUtilities.append(((1-protectionOdds[i])*aRewards[lam][i]) + (protectionOdds[i]*aPenalties[lam][i]))
    aStrat = argmax(expectedUtilities)
defenderSocialUtility = 0
for m in defenders:
    defenderSocialUtility += dStrat[m][aStrat] * (dRewards[m][aStrat] + dCosts[m][aStrat]) + (1-dStrat[m][aStrat]) * (dPenalties[m][aStrat] + dCosts[m][aStrat])
print("--- %s seconds ---" % (getTime() - start_time))
