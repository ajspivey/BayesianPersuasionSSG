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
    answer = digits
    if len(answer) < length:
        for i in range(length - len(answer)):
            answer.append(0)
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
    for defenderAssignment in s:
        if defenderAssignment == k:
            defended = True
    for defender in defenders:
        targetIndex = s[defender]
        utility += dCosts[defender][targetIndex]
        if defended == False:
            utility += dPenalties[defender][k]
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
        utility += dPenalties[m][a]
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

def createGraph(title, xLabel, yLabel, v1, v1Label, v2, v2Label, v3, v3Label, xStart=1):
    g = plt.figure()
    plt.plot(range(xStart, len(v3) + xStart), v3, 'r', label=f'{v3Label}')
    plt.plot(range(xStart, len(v2) + xStart), v2, 'b', label=f'{v2Label}')
    plt.plot(range(xStart, len(v1) + xStart), v1, 'g', label=f'{v1Label}')
    plt.title(f"{title}")
    plt.xlabel(f"{xLabel}")
    plt.ylabel(f"{yLabel}")
    plt.legend()
    plt.savefig(f"./{title}.png")
    return g

def solveBPAllowOverlap(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    placements = getPlacements(defenders, targetNum)
    attackerActions = list(range(targetNum))
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)
    model = Model('BayesianPersuasionSolverWithOverlap')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
    objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a,defenders, dCosts, dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] * aUtility(s,a,lam, aPenalties, aRewards) for s in placements]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam, aPenalties, aRewards) for s in placements]) for a in attackerActions for b in attackerActions if a != b for lam in aTypes], names=[f"c att {lam} suggested {a}, but goes to {b}" for a in attackerActions for b in attackerActions if a != b for lam in aTypes])
    model.add_constraints([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m, dRewards, dPenalties, dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) >= \
                           sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m, dRewards, dPenalties, dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) \
                           for m in defenders for d in range(targetNum) for e in range(targetNum) if d != e for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value()

def solveBPNoRequiredDefenderAssignment(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    _dRewards = dRewards.copy()
    _dPenalties = dPenalties.copy()
    _dCosts = dCosts.copy()
    for m in defenders:
        _dRewards[m].append(0)
        _dPenalties[m].append(0)
        _dCosts[m].append(0)
    placements = getPlacements(defenders, targetNum + 1)
    attackerActions = list(range(targetNum))
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)
    model = Model('BayesianPersuasionSolverWithoutRequiredAssignment')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
    objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a,defenders, _dCosts, _dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] * aUtility(s,a,lam, aPenalties, aRewards) for s in placements]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam, aPenalties, aRewards) for s in placements]) for a in attackerActions for b in attackerActions if a != b for lam in aTypes], names=[f"c att {lam} suggested {a}, but goes to {b}" for a in attackerActions for b in attackerActions if a != b for lam in aTypes])
    model.add_constraints([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) >= \
                           sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) \
                           for m in defenders for d in range(targetNum + 1) for e in range(targetNum + 1) if d != e for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value()

def solveBaseline(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    placements = getPlacements(defenders, targetNum)
    attackerActions = list(range(targetNum))
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)
    baselineUtility = 0
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
    return baselineUtility

def iterateTargets(targetNum, avgCount, bpOverbudget, bpNRDAOverbudget, baselineOverbudget):
    bpBelowBudget = True
    bpNRDABelowBudget = True
    baselineBelowBudget = True
    bpUtility = 0
    bpTime = 0
    bpNRDAUtility = 0
    bpNRDATime = 0
    baselineUtility = 0
    baselineTime = 0
    for _ in range(avgCount):
        print(f"set {_} of avgCount")
        # Generate a new game
        defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, targetNum)
        aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, targetNum)

        # BP Allow Overlap
        if not bpOverbudget:
            bpStart = getTime()
            bpUtility += solveBPAllowOverlap(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
            bpTime += getTime() - bpStart
        else:
            bpUtility = None
            bpTime = None

        # BP No Required Assignment
        if not bpNRDAOverbudget:
            bpNRDAStart = getTime()
            bpNRDAUtility += solveBPNoRequiredDefenderAssignment(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
            bpNRDATime += getTime() - bpNRDAStart
        else:
            bpNRDAUtility = None
            bpNRDATime = None

        # # Build the Baseline Model
        if not baselineOverbudget:
            baselineStart = getTime()
            baselineUtility += solveBPNoRequiredDefenderAssignment(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q)
            baselineTime += getTime() - baselineStart
        else:
            baselineUtility = None
            baselineTime = None
    if not bpOverbudget:
        bpUtility /= avgCount
        bpTime /= avgCount
    if not bpNRDAOverbudget:
        bpNRDAUtility /= avgCount
        bpNRDATime /= avgCount
    if not baselineOverbudget:
        baselineUtility /= avgCount
        baselineTime /= avgCount
    return bpUtility, bpTime, bpNRDAUtility, bpNRDATime, baselineUtility, baselineTime

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
timeBudget = 0.05 # 30 minutes
DEFENDERNUM = 2
ATTACKERNUM = 2
TARGETNUM = 7
avgCount = 40
M = 10000
defenderUtilities = []
solutionTimes = []
models = {}

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
avgBPUtils = []
avgBPTimes = []
avgBPNRDAUtils = []
avgBPNRDATimes = []
avgBaselineUtils = []
avgBaselineTimes = []
# Iterate over the targets
bpOver = False
bpNRDAOver = False
baselineOver = False
for _ in range(3, 51):
    print(f"Iteration {_} of {50} for targetCount")
    bpUtility, bpTime, bpNRDAUtility, bpNRDATime, baselineUtility, baselineTime = iterateTargets(targetNum=_, avgCount=avgCount, bpOverbudget=bpOver, bpNRDAOverbudget=bpNRDAOver, baselineOverbudget=baselineOver)
    if bpTime is not None and bpTime > timeBudget:
        bpOver = True
    if bpNRDATime is not None and bpNRDATime > timeBudget:
        bpNRDAOver = True
    if baselineTime is not None and baselineTime > timeBudget:
        baselineOver = True
    avgBPUtils.append(bpUtility)
    avgBPTimes.append(bpTime)
    avgBPNRDAUtils.append(bpNRDAUtility)
    avgBPNRDATimes.append(bpNRDATime)
    avgBaselineUtils.append(baselineUtility)
    avgBaselineTimes.append(baselineTime)
uGraph = createGraph("Average Utilities", "Number of Targets", "Utility", avgBPUtils, "BP Utility", avgBPNRDAUtils, "BPNRD Utility", avgBaselineUtils, "Baseline Utility", xStart = 3)
tGraph = createGraph("Average Runtimes", "Number of Targets", "Runtime", avgBPTimes, "Persuasion Scheme Time", avgBPNRDATimes, "BPNRD Time", avgBaselineTimes, "Baseline Time", xStart = 3)
# Iterate over the defenders
# Iterate over the attackers

plt.show()
