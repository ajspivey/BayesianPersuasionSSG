# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from docplex.mp.model import Model
from itertools import permutations
from functools import reduce
from operator import mul
import random
import time

randThing = random.Random()
# randThing.seed(0)

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def generateRandomDefenders(defenderNum, targetNum):
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
            dPenalties[m][i] = -1 * randThing.randint(1,50)
            dCosts[m][i] = -1 * randThing.randint(1,10)
    return defenders, dRewards, dPenalties, dCosts

def generateRandomAttackers(attackerNum, targetNum):
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
            aRewards[a][i] = 0
            aPenalties[a][i] = -1 * randThing.randint(1,50)
    return attackers, aRewards, aPenalties, q

def getPlacements():
    return list(permutations(defenders + ([-1 * x for x in range(targetNum - len(defenders))])))

def getLambdaPlacements():
    lambdaPlacements = []
    for aType in aTypes:
        for s in placements:
            lambdaPlacements.append((aType, s))
    return lambdaPlacements

def utilityDI(m,x,lam,i):
    u = x[i] * dRewards[m][i] + (1-x[i]) * dPenalties[m][i]
    return u

def utilityLamI(x,lam,i):
    u = x[i] * aPenalties[lam][i] + (1-x[i]) * aRewards[lam][i]
    return u

def probabilityProtected(dStrats, targetNum):
    protectionOdds = []
    strats = list(zip(*dStrats.values()))
    for probabilities in strats:
        protectionOdds.append(1-reduce(mul, [1-odd for odd in probabilities]))
    return protectionOdds

start_time = time.time()
# ==============================================================================
# GAME SETTINGS
# ==============================================================================
targetNum = 3
defenderNum = 2
attackerNum = 2
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
    model = Model('defenderStrategy')
    x = model.continuous_var_list(keys=targetNum, lb=0, ub=1, name="x")
    l = model.continuous_var_dict(keys=aTypes, lb=-model.infinity, name="UtilityLam")
    h = model.binary_var_dict(keys=[(lam, k) for lam in aTypes for k in range(targetNum)], lb=0, ub=1, name="h")
    ud = model.continuous_var_dict(keys=[lam for lam in aTypes], lb=-model.infinity, name="ud")
    objectiveFunction = sum([q[lam - 1] * ud[lam] for lam in aTypes])
    model.add_constraints([ud[lam] <= utilityDI(m,x,lam,k) + (1-h[(lam,k)]) * M for k in range(targetNum) for lam in aTypes])
    model.add_constraints([l[lam] <= utilityLamI(x,lam,k) + (1-h[(lam,k)]) * M for k in range(targetNum) for lam in aTypes])
    model.add_constraints([l[lam] >= utilityLamI(x,lam,k) for k in range(targetNum) for lam in aTypes])
    model.add_constraint(sum([x[i] for i in range(targetNum)]) == 1)
    model.add_constraints([sum([h[(lam,k)] for k in range(targetNum)]) == 1 for lam in aTypes])
    # Solve the problem
    model.maximize(objectiveFunction)
    model.solve()
    model.export("modelBaseline.lp")
    dStrat[m] = list([float(xVal) for xVal in x])
    models[m] = model
# Attacker response
aStrat = {}
protectionOdds = probabilityProtected(dStrat, targetNum)
for lam in aTypes:

print("--- %s seconds ---" % (time.time() - start_time))
