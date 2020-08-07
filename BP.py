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
randThing.seed(0)

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
            aRewards[a][i] = randThing.randint(1,50)
            aPenalties[a][i] = -1 * randThing.randint(1,50)
    return attackers, aRewards, aPenalties, q

def getPlacements():
    return list(permutations(defenders + ([0 * (targetNum - len(defenders))])))

def getLambdaPlacements():
    lambdaPlacements = []
    for aType in aTypes:
        for s in placements:
            lambdaPlacements.append((aType, s))
    return lambdaPlacements

def utilityK(s,k):
    if s[k] == 0: # Undefended
        utility = sum([penalty[k] for penalty in dPenalties.values()])
    else: # Defended
        utility = dRewards[s[k]][k]
    return utility

def utilityM(s,i,k,m):
    if i == k:                                  # Defended by defender m
        utility = dRewards[m][k]
    elif s[k] != 0 and s[k] != m and i != k:    # Defended by another defender
        utility = 0
    else:                                       # Undefended
        utility = dPenalties[m][k]
    return utility

start_time = getTime()
# ==============================================================================
# GAME SETTINGS
# ==============================================================================
targetNum = 3
defenderNum = 2
attackerNum = 2
M = 9999999
defenders, dRewards, dPenalties, dCosts = generateRandomDefenders(defenderNum, targetNum)
aTypes, aRewards, aPenalties, q = generateRandomAttackers(attackerNum, targetNum)

defenders = [1,2]
dRewards = {1:[0,0,0],2:[0,0,0]}
dPenalties = {1:[-1,-2,-3],2:[-3,-1,-1]}
dCosts = {1:[1,1,2],2:[1,1,1]}
# Attackers
aTypes = [1,2]
q = [0.7, 0.3]
aRewards = {1:[1,7,3], 2:[2,1,5]}
aPenalties = {1:[-2,-2,-2], 2:[-1,-1,-2]}

placements = getPlacements()
lambdaPlacements = getLambdaPlacements()

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
# Create the model and objective function
model = Model('BayesianPersuasionSolver')
z = model.continuous_var_dict(keys=lambdaPlacements, lb=-model.infinity, name="z")
w = model.continuous_var_dict(keys=lambdaPlacements, lb=0.0, ub=1, name="w")
y = model.continuous_var_dict(keys=[(lam,s,m,i) for lam in aTypes for s in placements for m in defenders for i in range(targetNum)], lb=-model.infinity, name="y")
h = model.binary_var_dict(keys=[(lam, k) for lam in aTypes for k in range(targetNum)], lb=0, ub=1, name="h")
v = model.continuous_var_dict(keys=aTypes, lb=-model.infinity, name="v")

objectiveFunction = sum([q[lam - 1] * sum([z[(lam,s)] for s in placements]) for lam in aTypes])

# Add the constraints
# Z constraints
model.add_constraints([z[(lam,s)] <= w[(lam,s)] * utilityK(s,k) + (1 - h[(lam, k)])*M for k in range(targetNum) for s in placements for lam in aTypes])
# Y constraints
model.add_constraints([y[(lam,s,m,i)] >= w[(lam,s)] * utilityM(s,i,k,m) - (1-h[(lam,k)]) * M for s in placements for i in range(targetNum) for m in defenders for lam in aTypes for k in range(targetNum)])
model.add_constraints([y[(lam,s,m,i)] <= w[(lam,s)] * utilityM(s,i,k,m) + (1-h[(lam,k)]) * M for s in placements for i in range(targetNum) for m in defenders for lam in aTypes for k in range(targetNum)])
model.add_constraints([sum(q[lam - 1] * sum([y[(lam,s,m,i)] for s in placements if s[i] == m]) for lam in aTypes) >= sum(q[lam - 1] * sum([y[(lam,s,m,j)] for s in placements if s[i] == m]) for lam in aTypes) for i in range(targetNum) for j in range(targetNum) if j != i for m in defenders])
# W constraints
model.add_constraints([sum([w[(lam,s)] for s in placements]) == 1.0 for lam in aTypes])
# V constraints
model.add_constraints([v[lam] >= sum([w[(lam,s)] for s in placements if s[i] != 0])*(aPenalties[lam][i] - aRewards[lam][i]) + aRewards[lam][i] for i in range(targetNum) for lam in aTypes])
model.add_constraints([v[lam] <= sum([w[(lam,s)] for s in placements if s[i] != 0])*(aPenalties[lam][i] - aRewards[lam][i]) + aRewards[lam][i] + (1 - h[(lam,i)])*M for i in range(targetNum) for lam in aTypes])
# H constraints -- already a binary variable
model.add_constraints([sum([h[(lam,k)] for k in range(targetNum)]) == 1 for lam in aTypes])

# Solve the problem
model.maximize(objectiveFunction)
model.solve()
model.export("model.lp")
print("--- %s seconds ---" % (getTime() - start_time))
print(model.get_solve_status())
print(model.solution.get_objective_value())
for k, v in w.items():
    print(k, float(v))
