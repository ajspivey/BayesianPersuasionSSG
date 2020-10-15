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

RAND = random.Random()
# RAND.seed(10)

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
            dPenalties[m].append(-1 * RAND.randint(1,penaltyCeiling))
            dCosts[m].append(-1 * RAND.randint(1,costCeiling))
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
            qVal = RAND.uniform(0,probability)
            probability -= qVal
            q.append(qVal)
        aRewards[a] = []
        aPenalties[a] = []
        for i in range(targetNum):
            aRewards[a].append(RAND.randint(1,rewardCeiling))
            aPenalties[a].append(-1 * RAND.randint(1,penaltyCeiling))
    return attackers, aRewards, aPenalties, q

def numberToBase(n, b, length):
    if n == 0:
        answer = [0] * length
        return tuple(answer)
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    answer = digits
    if len(answer) < length:
        for i in range(length - len(answer)):
            answer.append(0)
    return tuple(answer)

def getPlacements(defenders, targetNum):
    return [numberToBase(i,targetNum,len(defenders)) for i in range(targetNum ** len(defenders))]

def getOmegaKeys():
    omegaKeys = []
    for aType in aTypes:
        for s in placements:
            for a in attackerActions:
                omegaKeys.append((s, a, aType))
    return list(set(omegaKeys))

def defenderSocialUtility(s,k):
    utility = 0
    defended = False
    for defenderAssignment in s:
        if defenderAssignment == k:
            defended = True
    for defender in defenders:
        targetIndex = s[defender]
        utility += dCosts[defender][targetIndex]
        if defended == False:
            utility += dPenalties[defender][k]
    return utility

def utilityM(d,dm,a,m):
    utility = dCosts[m][d]
    defended = False
    for dIndex in range(len(dm)):
        if dIndex != m and dm[dIndex] == a:
            defended = True
    if d == a:
        defended = True
    if defended == False:
        utility += dPenalties[m][a]
    return utility

def aUtility(s,a,lam):
    utility = aRewards[lam][a]
    defended = False
    for targetIndex in s:
        if targetIndex == a:
            defended = True
    if defended == True:
        utility = aPenalties[lam][a]
    return utility

start_time = getTime()
# ==============================================================================
# GAME SETTINGS
# ==============================================================================
targetNum = 3
defenderNum = 2
attackerNum = 1
M = 1000
defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(defenderNum, targetNum)
dRewards = {0:[0,0,0], 1:[0,0,0]}
dPenalties = {0:[-8,-6,-11], 1:[-5,-17,-8]}
dCosts = {0:[-10,-9,-1], 1:[-8,-5,-7]}
aTypes, aRewards, aPenalties, q = generateRandomAttackers(attackerNum, targetNum)
aRewards = {0:[5,1,10]}
aPenalties = {0:[-17,-18,-11]}
placements = getPlacements(defenders, targetNum)
attackerActions = list(range(targetNum))
omegaKeys = getOmegaKeys()

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
# Create the model and objective function
model = Model('BayesianPersuasionSolver')
w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")

objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a) for s in placements for a in attackerActions]) for lam in aTypes])
# Constraints
model.add_constraints([sum([sum([w[(s,a,lam)] * aUtility(s,a,lam) for s in placements]) for a in attackerActions]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam) for s in placements for a in attackerActions]) for lam in aTypes for b in attackerActions], names=[f"attacker {lam} goes to {b}" for lam in aTypes for b in attackerActions])
model.add_constraints([sum([q[lam] * sum([sum([w[(dm,a,lam)] * utilityM(d,dm,a,m) for dm in placements if dm[m] == d for a in attackerActions]) for d in range(targetNum) ]) for lam in aTypes]) >= \
                        sum([ q[lam] * sum([ w[(dm,a,lam)] * utilityM(e,dm,a,m) for a in attackerActions for d in range(targetNum) for dm in placements if dm[m] == d]) for lam in aTypes])
                        for m in defenders for e in range(targetNum) for lam in aTypes], names=[f"defender {m} to {e}" for m in defenders for e in range(targetNum) for lam in aTypes])
model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
# Solve the problem
model.maximize(objectiveFunction)
model.solve()
model.export("relaxedModel.lp")
print("--- %s seconds ---" % (getTime() - start_time))
print(model.get_solve_status())
# for k, v in w.items():
#     print(k, float(v))

for _ in range(100):
    print(f"Run {_} of 100")
    defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(defenderNum, targetNum)
    aTypes, aRewards, aPenalties, q = generateRandomAttackers(attackerNum, targetNum)
    placements = getPlacements(defenders, targetNum)
    attackerActions = list(range(targetNum))
    omegaKeys = getOmegaKeys()
    model = Model('BayesianPersuasionSolver')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")

    objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a) for s in placements for a in attackerActions]) for lam in aTypes])
    # Constraints
    model.add_constraints([sum([sum([w[(s,a,lam)] * aUtility(s,a,lam) for s in placements]) for a in attackerActions]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam) for s in placements for a in attackerActions]) for lam in aTypes for b in attackerActions], names=[f"attacker {lam} goes to {b}" for lam in aTypes for b in attackerActions])
    model.add_constraints([sum([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m) for a in attackerActions for dm in placements if dm[m] == d])]) for d in range(targetNum)]) >= \
                           sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m) for a in attackerActions for d in range(targetNum) for dm in placements if dm[m] == d])]) \
                           for m in defenders for e in range(targetNum) for lam in aTypes], names=[f"defender {m} ignores all suggestions and goes to {e} every time" for m in defenders for e in range(targetNum) for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
    # Solve the problem
    model.maximize(objectiveFunction)
    model.solve()
