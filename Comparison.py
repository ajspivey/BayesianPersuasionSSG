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
        dRewards[m] = {}
        dPenalties[m] = {}
        dCosts[m] = {}
        for i in range(targetNum):
            dRewards[m][i] = 0
            dPenalties[m][i] = -1 * randThing.randint(1,penaltyCeiling)
            dCosts[m][i] = -1 * randThing.randint(0,costCeiling)
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
        aRewards[a] = {}
        aPenalties[a] = {}
        for i in range(targetNum):
            aRewards[a][i] = randThing.randint(1,rewardCeiling)
            aPenalties[a][i] = -1 * randThing.randint(1,penaltyCeiling)
    return attackers, aRewards, aPenalties, q

def getPlacements(defenders):
    return list(set(permutations(defenders + ([-1] * (targetNum - len(defenders))))))

def getLambdaPlacements(aTypes, placements):
    lambdaPlacements = []
    for aType in aTypes:
        for s in placements:
            lambdaPlacements.append((aType, s))
    return lambdaPlacements

def utilityDI(m,x,lam,i,dRewards,dPenalties):
    return x[i] * dRewards[m][i] + (1-x[i]) * dPenalties[m][i]

def utilityLamI(x,lam,i,aRewards,aPenalties):
    return x[i] * aPenalties[lam][i] + (1-x[i]) * aRewards[lam][i]

def probabilityProtected(dStrats, targetNum):
    protectionOdds = []
    strats = list(zip(*dStrats.values()))
    for probabilities in strats:
        protectionOdds.append(1-reduce(mul, [1-odd for odd in probabilities]))
    return protectionOdds

def utilityK(s,k,dPenalties):
    utility = 0 # Defended
    if s[k] == -1: # Undefended
        utility = sum([penalty[k] for penalty in dPenalties.values()])
    return utility

def utilityM(s,i,k,m,dPenalties):
    utility = 0 # Defended
    if s[k] == -1 or (s[k] == m and i != k): # Undefended
        utility = sum([penalty[k] for penalty in dPenalties.values()])
    return utility

def getAvgUtilitiesAndTimes(targetNum, avgCount=10):
    bpUtility = 0
    bpTime = 0
    baselineUtility = 0
    baselineTime = 0
    for _ in range(avgCount):
        print(f"set {_} of avgCount")
        # Generate a new game
        defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, targetNum)
        aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, targetNum)
        placements = getPlacements(defenders)
        lambdaPlacements = getLambdaPlacements(aTypes, placements)
        # Build the BP Model
        bpStart = getTime()
        model = Model('BayesianPersuasionSolver')
        z = model.continuous_var_dict(keys=lambdaPlacements, lb=-model.infinity, name="z")
        w = model.continuous_var_dict(keys=lambdaPlacements, lb=0, ub=1, name="w")
        y = model.continuous_var_dict(keys=[(lam,s,m,i) for lam in aTypes for s in placements for m in defenders for i in range(targetNum)], lb=-model.infinity, name="y")
        h = model.binary_var_dict(keys=[(lam, k) for lam in aTypes for k in range(targetNum)], lb=0, ub=1, name="h")
        v = model.continuous_var_dict(keys=aTypes, lb=-model.infinity, name="v")
        objectiveFunction = sum([q[lam - 1] * sum([z[(lam,s)] for s in placements]) for lam in aTypes])
        model.add_constraints([z[(lam,s)] <= w[(lam,s)] * utilityK(s,k,dPenalties) + (1 - h[(lam, k)])*M for k in range(targetNum) for s in placements for lam in aTypes])
        model.add_constraints([sum(q[lam - 1] * sum([y[(lam,s,m,i)] for s in placements if s[i] == m]) for lam in aTypes) >= sum(q[lam - 1] * sum([y[(lam,s,m,j)] for s in placements if s[i] == m]) for lam in aTypes) for i in range(targetNum) for j in range(targetNum) if j != i for m in defenders])
        model.add_constraints([y[(lam,s,m,i)] >= w[(lam,s)]  * utilityM(s,i,k,m,dPenalties) - (1-h[(lam,k)]) * M for s in placements for lam in aTypes for i in range(targetNum) for k in range(targetNum) for m in defenders])
        model.add_constraints([y[(lam,s,m,i)] <= w[(lam,s)]  * utilityM(s,i,k,m,dPenalties) + (1-h[(lam,k)]) * M for s in placements for lam in aTypes for i in range(targetNum) for k in range(targetNum) for m in defenders])
        model.add_constraints([sum([w[(lam,s)] for s in placements]) == 1 for lam in aTypes])
        model.add_constraints([v[lam] >= sum([w[(lam,s)] for s in placements if s[i] != -1])*(aPenalties[lam][i] - aRewards[lam][i]) + aRewards[lam][i] for i in range(targetNum) for lam in aTypes])
        model.add_constraints([v[lam] <= sum([w[(lam,s)] for s in placements if s[i] != -1])*(aPenalties[lam][i] - aRewards[lam][i]) + aRewards[lam][i] + (1 - h[(lam,i)])*M for i in range(targetNum) for lam in aTypes])
        model.add_constraints([sum([h[(lam,k)] for k in range(targetNum)]) == 1 for lam in aTypes])
        model.maximize(objectiveFunction)
        model.solve()
        bpUtility += model.solution.get_objective_value()
        bpTime += getTime() - bpStart
        # Build the Baseline Model
        baselineStart = getTime()
        dStrat = {}
        models2 = {}
        for m in defenders:
            model2 = Model('defenderStrategy')
            x = model2.continuous_var_list(keys=targetNum, lb=0, ub=1, name="x")
            l = model2.continuous_var_dict(keys=aTypes, lb=-model2.infinity, name="UtilityLam")
            h = model2.binary_var_dict(keys=[(lam, k) for lam in aTypes for k in range(targetNum)], lb=0, ub=1, name="h")
            ud = model2.continuous_var_dict(keys=[lam for lam in aTypes], lb=-model2.infinity, name="ud")
            objectiveFunction2 = sum([q[lam - 1] * ud[lam] for lam in aTypes])
            model2.add_constraints([ud[lam] <= utilityDI(m,x,lam,k,dRewards,dPenalties) + (1-h[(lam,k)]) * M for k in range(targetNum) for lam in aTypes])
            model2.add_constraints([l[lam] <= utilityLamI(x,lam,k,aRewards,aPenalties) + (1-h[(lam,k)]) * M for k in range(targetNum) for lam in aTypes])
            model2.add_constraints([l[lam] >= utilityLamI(x,lam,k,aRewards,aPenalties) for k in range(targetNum) for lam in aTypes])
            model2.add_constraint(sum([x[i] for i in range(targetNum)]) == 1)
            model2.add_constraints([sum([h[(lam,k)] for k in range(targetNum)]) == 1 for lam in aTypes])
            # Solve the problem
            model2.maximize(objectiveFunction2)
            model2.solve()
            model2.export("modelBaseline.lp")
            dStrat[m] = list([float(xVal) for xVal in x])
            models2[m] = model2
        baselineUtility += sum([model2.solution.get_objective_value() for model2 in models2.values()])
        baselineTime += getTime() - baselineStart
    bpUtility /= avgCount
    bpTime /= avgCount
    baselineUtility /= avgCount
    baselineTime /= avgCount
    return bpUtility, bpTime, baselineUtility, baselineTime

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
targetNums = 10
DEFENDERNUM = 2
ATTACKERNUM = 2
M = 9999999
defenderUtilities = []
solutionTimes = []
models = []

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
avgUtils = []
avgTimes = []
for targetNum in range(3, targetNums + 1):
    print(f"targetSize {targetNum} of {targetNums}")
    bpUtility, bpTime, baselineUtility, baselineTime = getAvgUtilitiesAndTimes(targetNum)
    avgUtils.append((bpUtility, baselineUtility))
    avgTimes.append((bpTime, baselineTime))

print(avgUtils)
print(avgTimes)
