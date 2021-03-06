from docplex.mp.model import Model
from itertools import permutations
from functools import reduce
from operator import mul
from numpy import argmax
from time import time as getTime
import matplotlib.pyplot as plt
import random
import copy

# ==============================================================================
# FUNCTIONS
# ==============================================================================
# ------------------------------------------------------------------------------
def generateRandomDefenders(defenderNum, targetNum, rewardCeiling=20, penaltyCeiling=20, costCeiling=10):
    """Generates defenders with random rewards, penalties, and costs, given a number of targets"""
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

# ------------------------------------------------------------------------------
def generateRandomAttackers(attackerNum, targetNum, rewardCeiling=20, penaltyCeiling=20):
    """Generates defenders with random rewards and penalties given a number of targets"""
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

# ------------------------------------------------------------------------------
def numberToBase(n, b, length):
    """Converts a number n into base b with the highest digits on the right.
    Digits higher than 9 are represented with base 10 digits (e.g. 10 in base 11 becomes 10,0)
    This is used to generate all potential placements for defenders using getPlacements"""
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

# ------------------------------------------------------------------------------
def getPlacements(defenders, targetNum):
    """Generates all potential placements for a number of defenders and targets
    by essentially counting up to the maximum number of placements in the base
    of the number of targets (e.g. counting up to 9 in base 3 with two digits gives
    you all placements of 2 defenders over three targets)"""
    return [numberToBase(i,targetNum,len(defenders)) for i in range(targetNum ** len(defenders))]

# ------------------------------------------------------------------------------
def getOmegaKeys(aTypes, placements, attackerActions):
    """Pairs all placements with all possible values of lambda to generate variable
    keys for omega ( w(s,a|lambda) )"""
    omegaKeys = []
    for aType in aTypes:
        for s in placements:
            for a in attackerActions:
                omegaKeys.append((s, a, aType))
    return list(set(omegaKeys))

# ------------------------------------------------------------------------------
def defenderSocialUtility(s,a,defenders, dRewards, dCosts, dPenalties):
    """Calculates the defender's social utility (Sum of the utility of all defenders)
    given the costs and penalties for the defenders as well as the placements and
    attacked target"""
    utility = 0
    defended = False
    for defender in defenders:
        targetIndex = s[defender]
        utility += dCosts[defender][targetIndex]
        if targetIndex == a:
            defended = True
            utility += dRewards[defender][targetIndex]
    if defended == False:
        utility += sum([dPenalties[defender][a] for defender in defenders])
    return utility

# ------------------------------------------------------------------------------
def utilityM(d,dm,a,m, dRewards, dPenalties, dCosts):
    """Calculates the utility for an individual defender if they go to target
    d and target a is attacked. all other defenders follow suggestion dm"""
    utility = dCosts[m][d]
    defended = False
    for dIndex in range(len(dm)):
        if dIndex != m and dm[dIndex] == a:
            defended = True
    if d == a:
        defended = True
        utility += dRewards[m][d]
    if defended == False:
        utility += dPenalties[m][a]
    return utility

# ------------------------------------------------------------------------------
def aUtility(s,a,lam, aPenalties, aRewards):
    """Calculates the utility for an attacker lambda when the defenders follow
    plaement s and the attacker attacks target a"""
    if a in s:
        return aPenalties[lam][a]
    else:
        return aRewards[lam][a]

# ------------------------------------------------------------------------------
def getLambdaPlacements():
    """Used in the same manner as omegaKeys to construct variable keys for the baseline"""
    lambdaPlacements = []
    for aType in aTypes:
        for s in placements:
            lambdaPlacements.append((aType, s))
    return lambdaPlacements

# ------------------------------------------------------------------------------
def utilityDI(m,x,i,dRewards,dPenalties,dCosts):
    """Calculates the utility of an individual defender m for mixed strategy x
    when target i is attacked"""
    utility = x[i] * (dRewards[m][i] + dCosts[m][i]) + (1-x[i]) * (dPenalties[m][i] + dCosts[m][i])
    return utility

# ------------------------------------------------------------------------------
def utilityLamI(x,lam,i,aRewards,aPenalties):
    """Calculates the utility of an individual attacker lam for mixed strategy x
    when target i is attacked"""
    utility = x[i] * aPenalties[lam][i] + (1-x[i]) * aRewards[lam][i]
    return utility

# ------------------------------------------------------------------------------
def probabilityProtected(dStrats, targetNum):
    """Given a mixed strategy for each defender, calculates the chance that each
    target is protected by a defender. Returns a list of probabilities"""
    protectionOdds = []
    strats = list(zip(*dStrats.values()))
    for probabilities in strats:
        protectionOdds.append(1-reduce(mul, [1-odd for odd in probabilities]))
    return protectionOdds

# ------------------------------------------------------------------------------
def createGraph(title, xLabel, yLabel, gameTypes, gameData, xStart=1):
    """Given a number of labels and values, creates a graph comparing performance"""
    g = plt.figure()
    for gameIndex in range(len(gameTypes)):
        label = gameTypes[gameIndex][1]
        color = gameTypes[gameIndex][2]
        data = gameData[label]
        plt.plot(range(xStart, len(data) + xStart), data, f"{color}", label=f'{label} {yLabel}')
    plt.title(f"{title}")
    plt.xlabel(f"{xLabel}")
    plt.ylabel(f"{yLabel}")
    plt.legend()
    plt.savefig(f"./Comparisons/{title}.png")
    return g
