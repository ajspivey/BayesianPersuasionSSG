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

# def getLambdaPlacements(aTypes, placements):
#     lambdaPlacements = []
#     for aType in aTypes:
#         for s in placements:
#             lambdaPlacements.append((aType, s))
#     return lambdaPlacements
#
# def utilityDI(m,x,lam,i,dRewards,dPenalties):
#     return x[i] * dRewards[m][i] + (1-x[i]) * dPenalties[m][i]
#
# def utilityLamI(x,lam,i,aRewards,aPenalties):
#     return x[i] * aPenalties[lam][i] + (1-x[i]) * aRewards[lam][i]
#
# def probabilityProtected(dStrats, targetNum):
#     protectionOdds = []
#     strats = list(zip(*dStrats.values()))
#     for probabilities in strats:
#         protectionOdds.append(1-reduce(mul, [1-odd for odd in probabilities]))
#     return protectionOdds
#
# def utilityK(s,k,dPenalties):
#     utility = 0 # Defended
#     if s[k] == -1: # Undefended
#         utility = sum([penalty[k] for penalty in dPenalties.values()])
#     return utility

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
        # print("\n\n\n\n\n\n\n\n\n")
        # Generate a new game
        defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(DEFENDERNUM, targetNum)
        aTypes, aRewards, aPenalties, q = generateRandomAttackers(ATTACKERNUM, targetNum)
        # dCosts = {0: [0, 0, 0], 1: [0, 0, 0]}
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
        try:
            bpUtility += model.solution.get_objective_value()
            # print(f"BP: {model.solution.get_objective_value()}")
        except Exception as err:
            print(f"Defender rewards: {dRewards}")
            print(f"Defender penalties: {dPenalties}")
            print(f"Defender costs: {dCosts}")
            print(f"Attacker rewards: {aRewards}")
            print(f"Attacker Penalties: {aPenalties}")
            print(f"Q: {q}")
            model.export("badModel.lp")
            print(model.get_solve_status())
            asdf
        bpTime += getTime() - bpStart
        # Build the Baseline Model
        # baselineStart = getTime()
        # dStrat = {}
        # models2 = {}
        # for m in defenders:
        #     model2 = Model('defenderStrategy')
        #     x = model2.continuous_var_list(keys=targetNum, lb=0, ub=1, name="x")
        #     l = model2.continuous_var_dict(keys=aTypes, lb=-model2.infinity, name="UtilityLam")
        #     h = model2.binary_var_dict(keys=[(lam, k) for lam in aTypes for k in range(targetNum)], lb=0, ub=1, name="h")
        #     ud = model2.continuous_var_dict(keys=[lam for lam in aTypes], lb=-model2.infinity, name="ud")
        #     objectiveFunction2 = sum([q[lam - 1] * ud[lam] for lam in aTypes])
        #     model2.add_constraints([ud[lam] <= utilityDI(m,x,lam,k,dRewards,dPenalties) + (1-h[(lam,k)]) * M for k in range(targetNum) for lam in aTypes])
        #     model2.add_constraints([l[lam] <= utilityLamI(x,lam,k,aRewards,aPenalties) + (1-h[(lam,k)]) * M for k in range(targetNum) for lam in aTypes])
        #     model2.add_constraints([l[lam] >= utilityLamI(x,lam,k,aRewards,aPenalties) for k in range(targetNum) for lam in aTypes])
        #     model2.add_constraint(sum([x[i] for i in range(targetNum)]) == 1)
        #     model2.add_constraints([sum([h[(lam,k)] for k in range(targetNum)]) == 1 for lam in aTypes])
        #     # Solve the problem
        #     model2.maximize(objectiveFunction2)
        #     model2.solve()
        #     model2.export("modelBaseline.lp")
        #     dStrat[m] = list([float(xVal) for xVal in x])
        #     models2[m] = model2
        # baselineUtility += sum([model2.solution.get_objective_value() for model2 in models2.values()])
        # print(f"Baseline: {sum([model2.solution.get_objective_value() for model2 in models2.values()])}")
        # print(f"Model values: {[model2.solution.get_objective_value() for model2 in models2.values()]}")
        # baselineTime += getTime() - baselineStart
    bpUtility /= avgCount
    bpTime /= avgCount
    # baselineUtility /= avgCount
    # baselineTime /= avgCount
    return bpUtility, bpTime, baselineUtility, baselineTime

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
targetNums = 5
avgCount = 1000
DEFENDERNUM = 2
ATTACKERNUM = 1
M = 10000
defenderUtilities = []
solutionTimes = []
models = []

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
avgBPUtils = []
avgBPTimes = []
avgBaselineUtils = []
avgBaselineTimes = []
for targetNum in range(3, targetNums + 1):
    print(f"targetSize {targetNum} of {targetNums}")
    bpUtility, bpTime, baselineUtility, baselineTime = getAvgUtilitiesAndTimes(targetNum, avgCount)
    avgBPUtils.append(bpUtility)
    avgBPTimes.append(bpTime)
    avgBaselineUtils.append(baselineUtility)
    avgBaselineTimes.append(baselineTime)

uGraph = createGraph("Average Utilities", "Number of Targets", "Utility", avgBPUtils, "Persuasion Scheme Utility", avgBaselineUtils, "Baseline Utility")
tGraph = createGraph("Average Runtimes", "Number of Targets", "Runtime", avgBPTimes, "Persuasion Scheme Time", avgBaselineTimes, "Baseline Time")
plt.show()
