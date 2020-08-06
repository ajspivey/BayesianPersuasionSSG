# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from docplex.mp.model import Model
from itertools import permutations
import time

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def getPlacements():
    return list(permutations(defenders + ([-1 * x for x in range(targetNum - len(defenders))])))

def getLambdaPlacements():
    lambdaPlacements = []
    for aType in aTypes:
        for s in placements:
            lambdaPlacements.append((aType, s))
    return lambdaPlacements

def utilityK(s,k):
    utility = 0 # Defended
    if s[k] < 0: # Undefended
        utility = sum([penalty[k] for penalty in dPenalties.values()])
    return utility

def utilityM(s,i,k,m):
    utility = 0 # Defended
    if s[k] < 0 or (s[k] == m and i != k): # Undefended
        utility = dPenalties[m][k]
    return utility

start_time = time.time()
# ==============================================================================
# GAME SETTINGS
# ==============================================================================
# Game and optimization
targetNum = 10
M = 9999
# Attackers
aTypes = [1,2]
aRewards = {1:[1,7,3,4,2,1,2,3,4,5], 2:[2,1,5,9,1,5,4,3,2,1]}
aPenalties = {1:[-2,-2,-2,-3,-2,-2,-2,-2,-3,-2], 2:[-1,-1,-2,-4,-1,-1,-2,-3,-4,-5]}
q = [0.7, 0.3]
# Defenders
defenders = [1,2]
dRewards = {1:[0,0,0,0,0,0,0,0,0,0],2:[0,0,0,0,0,0,0,0,0,0]}
dPenalties = {1:[-1,-2,-3,-1,-2,-1,-2,-3,-4,-5],2:[-3,-1,-1,-5,-4,-5,-4,-3,-2,-1]}
dCosts = {1:[1,1,2,1,1,1,2,1,2,1],2:[1,1,1,2,2,1,1,1,2,2]}
# Create the list of lambda placement keys -- all possible assignments, with each defender
# type attached.
placements = getPlacements()
lambdaPlacements = getLambdaPlacements()

# ==============================================================================
# LP Definition & Constraints
# ==============================================================================
# Create the model and objective function
model = Model('BayesianPersuasionSolver')
z = model.continuous_var_dict(keys=lambdaPlacements, lb=-model.infinity, name="z")
w = model.continuous_var_dict(keys=lambdaPlacements, lb=0, ub=1, name="w")
y = model.continuous_var_dict(keys=[(lam,s,m,i) for lam in aTypes for s in placements for m in defenders for i in range(targetNum)], lb=-model.infinity, name="y")
h = model.binary_var_dict(keys=[(lam, k) for lam in aTypes for k in range(targetNum)], lb=0, ub=1, name="h")
v = model.continuous_var_dict(keys=aTypes, lb=-model.infinity, name="v")

objectiveFunction = sum([q[lam - 1] * sum([z[(lam,s)] for s in placements]) for lam in aTypes])

# Add the constraints
# Z constraints
model.add_constraints([z[(lam,s)] <= w[(lam,s)] * utilityK(s,k) + (1 - h[(lam, k)])*M for k in range(targetNum) for s in placements for lam in aTypes])
# Y constraints
model.add_constraints([sum(q[lam - 1] * sum([y[(lam,s,m,i)] for s in placements if s[i] == m]) for lam in aTypes) >= sum(q[lam - 1] * sum([y[(lam,s,m,j)] for s in placements if s[i] == m]) for lam in aTypes) for i in range(targetNum) for j in range(targetNum) if j != i for m in defenders])
model.add_constraints([y[(lam,s,m,i)] >= w[(lam,s)]  * utilityM(s,i,k,m) - (1-h[(lam,k)]) * M for s in placements for lam in aTypes for i in range(targetNum) for k in range(targetNum) for m in defenders])
model.add_constraints([y[(lam,s,m,i)] <= w[(lam,s)]  * utilityM(s,i,k,m) + (1-h[(lam,k)]) * M for s in placements for lam in aTypes for i in range(targetNum) for k in range(targetNum) for m in defenders])
# W constraints
model.add_constraints([sum([w[(lam,s)] for s in placements]) == 1 for lam in aTypes])
# V constraints
model.add_constraints([v[lam] >= sum([w[(lam,s)] for s in placements if s[i] != -1])*(aPenalties[lam][i] - aRewards[lam][i]) + aRewards[lam][i] for i in range(targetNum) for lam in aTypes])
model.add_constraints([v[lam] <= sum([w[(lam,s)] for s in placements if s[i] != -1])*(aPenalties[lam][i] - aRewards[lam][i]) + aRewards[lam][i] + (1 - h[(lam,i)])*M for i in range(targetNum) for lam in aTypes])
# H constraints -- already a binary variable
model.add_constraints([sum([h[(lam,k)] for k in range(targetNum)]) == 1 for lam in aTypes])

# Solve the problem
model.maximize(objectiveFunction)
model.solve()
model.export("model.lp")
print("--- %s seconds ---" % (time.time() - start_time))
print(model.get_solve_status())
print(model.solution.get_objective_value())
print(list([float(wVal) for wVal in w.values()]))
