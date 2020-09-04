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

randThing = random.Random()
# randThing.seed(1)

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
            dPenalties[m].append(-1 * randThing.randint(1,penaltyCeiling))
            dCosts[m].append(-1 * randThing.randint(1,costCeiling))
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
            qVal = randThing.uniform(0,probability)
            probability -= qVal
            q.append(qVal)
        aRewards[a] = []
        aPenalties[a] = []
        for i in range(targetNum):
            aRewards[a].append(randThing.randint(1,rewardCeiling))
            aPenalties[a].append(-1 * randThing.randint(1,penaltyCeiling))
    return attackers, aRewards, aPenalties, q

def numberToBase(n, b, length):
    if n == 0:
        answer = [0] * length
        return tuple(answer)
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    answer = digits[::-1]
    if len(answer) < length:
        for i in range(length - len(answer)):
            answer.insert(0,0)
    return tuple(answer)

def getPlacements(defenders, targetNum):
    return [numberToBase(i,targetNum,len(defenders)) for i in range(targetNum ** len(defenders))]

def getOmegaKeys(aTypes, placements, attackerActions):
    omegaKeys = []
    for aType in aTypes:
        for s in placements:
            for a in attackerActions:
                omegaKeys.append((s, a, aType))
    return list(set(omegaKeys))

def defenderSocialUtility(s,k,defenders, dCosts, dPenalties):
    utility = 0
    defended = False
    for targetIndex in s:
        if targetIndex == k:
            defended = True
    for defender in defenders:
        targetIndex = s[defender]
        utility += dCosts[defender][targetIndex]
        if defended == False:
            utility += dPenalties[defender][targetIndex]
    # print(f"D social for s:{s}, a:{k}, = {utility}")
    return utility

def utilityM(d,dm,a,m, dRewards, dPenalties, dCosts):
    utility = dCosts[m][d]
    defended = False
    for dIndex in range(len(dm)):
        if dIndex != m and dm[dIndex] == a:
            defended = True
    if d == a:
        defended = True
    if defended == False:
        utility += dPenalties[m][d]
    # print(f"M utility for d:{d}, dm:{dm}, a:{a}, m:{m} = {utility}")
    return utility

def aUtility(s,a,lam, aPenalties, aRewards):
    utility = aRewards[lam][a]
    defended = False
    for targetIndex in s:
        if targetIndex == a:
            defended = True
    if defended == True:
        utility = aPenalties[lam][a]
    # print(f"A utility for s:{s}, a:{a} = {utility}")
    return utility

def getLambdaPlacements():
    lambdaPlacements = []
    for aType in aTypes:
        for s in placements:
            lambdaPlacements.append((aType, s))
    return lambdaPlacements

def utilityDI(m,x,i,dRewards,dPenalties,dCosts):
    utility = x[i] * (dRewards[m][i] + dCosts[m][i]) + (1-x[i]) * (dPenalties[m][i] + dCosts[m][i])
    return utility

def utilityLamI(x,lam,i,aRewards,aPenalties):
    utility = x[i] * aPenalties[lam][i] + (1-x[i]) * aRewards[lam][i]
    return utility

def probabilityProtected(dStrats, targetNum):
    protectionOdds = []
    strats = list(zip(*dStrats.values()))
    for probabilities in strats:
        protectionOdds.append(1-reduce(mul, [1-odd for odd in probabilities]))
    return protectionOdds

def createGraph(title, xLabel, yLabel, v1, v1Label, v2, v2Label):
    g = plt.figure()
    plt.plot(range(3, len(v1) + 3), v2, 'r', label=f'{v2Label}')
    plt.plot(range(3, len(v1) + 3), v1, 'g', label=f'{v1Label}')
    plt.title(f"{title}")
    plt.xlabel(f"{xLabel}")
    plt.ylabel(f"{yLabel}")
    plt.legend()
    plt.savefig(f"./{title}.png")
    return g

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
        placements = getPlacements(defenders, targetNum)
        attackerActions = list(range(targetNum))
        omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)
        # Build the BP Model
        bpStart = getTime()
        model = Model('BayesianPersuasionSolver')
        w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
        objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a,defenders,dCosts,dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])
        model.add_constraints([sum([w[(s,a,lam)] * aUtility(s,a,lam,aPenalties,aRewards) for s in placements]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam,aPenalties,aRewards) for s in placements]) for a in attackerActions for b in attackerActions if a != b for lam in aTypes], names=[f"c att {lam} suggested {a}, but goes to {b}" for a in attackerActions for b in attackerActions if a != b for lam in aTypes])
        model.add_constraints([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m,dRewards, dPenalties, dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) >= \
                               sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m,dRewards, dPenalties, dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) \
                               for m in defenders for d in range(targetNum) for e in range(targetNum) if d != e for lam in aTypes], names=[f"defender {m} suggested {d} but went to {e} vs. attacker {lam}" for m in defenders for d in range(targetNum) for e in range(targetNum) if d != e for lam in aTypes])
        model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes], names=[f"Probabilities must sum to 1 for attacker type {lam}" for lam in aTypes])
        model.maximize(objectiveFunction)
        model.solve()
        bpUtility += model.solution.get_objective_value()
        bpTime += getTime() - bpStart
        # Build the Baseline Model
        baselineStart = getTime()
        dStrat = {}
        models2 = {}
        for m in defenders:
            model2 = Model(f"defenderStrategy{m}")
            x = model2.continuous_var_list(keys=targetNum, lb=0, ub=1, name=f"x{m}")
            h = model2.binary_var_dict(keys=[(lam, k) for lam in aTypes for k in range(targetNum)], lb=0, ub=1, name=f"h{m}")
            ul = model2.continuous_var_dict(keys=aTypes, lb=-model2.infinity, name=f"ua{m}")
            ud = model2.continuous_var_dict(keys=[lam for lam in aTypes], lb=-model2.infinity, name=f"ud{m}")
            objectiveFunction = sum([q[lam] * ud[lam] for lam in aTypes])
            model2.add_constraints([ud[lam] <= utilityDI(m,x,i,dRewards,dPenalties,dCosts) + (1-h[(lam,i)]) * M for i in range(targetNum) for lam in aTypes], names=[f"defender utility for lam {lam}, i {i}" for i in range(targetNum) for lam in aTypes])
            model2.add_constraints([ul[lam] <= utilityLamI(x,lam,i,aRewards,aPenalties) + (1-h[(lam,i)]) * M for i in range(targetNum) for lam in aTypes], names=[f"lam {lam} utility leq for i {i}" for i in range(targetNum) for lam in aTypes])
            model2.add_constraints([ul[lam] >= utilityLamI(x,lam,i,aRewards,aPenalties) for i in range(targetNum) for lam in aTypes], names=[f"lam {lam} utility geq, for i {i}" for i in range(targetNum) for lam in aTypes])
            model2.add_constraints([sum([h[(lam,i)] for i in range(targetNum)]) == 1 for lam in aTypes], names=[f"h sum is 1 for lam {lam}" for lam in aTypes])
            model2.add_constraint(sum([x[i] for i in range(targetNum)]) == 1)
            # Solve the problem
            model2.maximize(objectiveFunction)
            model2.solve()
            model2.export(f"modelBaseline{m}.lp")
            dStrat[m] = list([float(xVal) for xVal in x])
            models[m] = model2
        # Attacker response
        aStrat = 0
        protectionOdds = probabilityProtected(dStrat, targetNum)
        for lam in aTypes:
            expectedUtilities = []
            for i in range(targetNum):
                expectedUtilities.append(((1-protectionOdds[i])*aRewards[lam][i]) + (protectionOdds[i]*aPenalties[lam][i]))
            aStrat = argmax(expectedUtilities)
        for m in defenders:
            baselineUtility += dStrat[m][aStrat] * (dRewards[m][aStrat] + dCosts[m][aStrat]) + (1-dStrat[m][aStrat]) * (dPenalties[m][aStrat] + dCosts[m][aStrat])
        baselineTime += getTime() - baselineStart
    bpUtility /= avgCount
    bpTime /= avgCount
    baselineUtility /= avgCount
    baselineTime /= avgCount
    return bpUtility, bpTime, baselineUtility, baselineTime

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
targetFloor = 3
targetCeiling = 7
avgCount = 100
DEFENDERNUM = 2
ATTACKERNUM = 1
M = 10000
defenderUtilities = []
solutionTimes = []
models = {}

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
avgBPUtils = []
avgBPTimes = []
avgBaselineUtils = []
avgBaselineTimes = []
for targetNum in range(targetFloor, targetCeiling + 1):
    print(f"targetSize {targetNum} of {targetCeiling}")
    bpUtility, bpTime, baselineUtility, baselineTime = getAvgUtilitiesAndTimes(targetNum, avgCount)
    avgBPUtils.append(bpUtility)
    avgBPTimes.append(bpTime)
    avgBaselineUtils.append(baselineUtility)
    avgBaselineTimes.append(baselineTime)

uGraph = createGraph("Average Utilities", "Number of Targets", "Utility", avgBPUtils, "Persuasion Scheme Utility", avgBaselineUtils, "Baseline Utility")
tGraph = createGraph("Average Runtimes", "Number of Targets", "Runtime", avgBPTimes, "Persuasion Scheme Time", avgBaselineTimes, "Baseline Time")
plt.show()
