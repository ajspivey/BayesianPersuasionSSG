# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from docplex.mp.model import Model
from itertools import permutations

def getPlacements():
    return list(permutations(defenders + ([0] * (targetNum - len(defenders)))))

def getLambdaPlacements():
    lambdaPlacements = []
    for aType in aTypes:
        for s in placements:
            lambdaPlacements.append((aType, s))
    return lambdaPlacements

# Game and optimization
targetNum = 3
m = 1000
# Attackers
aTypes = [1,2]
aRewards = {1:[1,7,3], 2:[2,1,5]}
aPenalties = {1:[2,2,2], 2:[1,1,2]}
q = [0.7, 0.3]
# Defenders
defenders = [1,2]
dRewards = {1:[0,0,0],2:[0,0,0]}
dPenalties = {1:[1,2,3],2:[3,1,1]}
dCosts = {1:[1,1,2],2:[1,1,1]}
# Create the list of lambda placement keys -- all possible assignments, with each defender
# type attached.
placements = getPlacements()
lambdaPlacements = getLambdaPlacements()

# Create the model and objective function
model = Model('BayesianPersuasionSolver')
z = model.continuous_var_dict(keys=lambdaPlacements, ub=2)
objectiveFunction = sum([q[lam - 1] * sum([z[(lam,s)] for s in placements]) for lam in aTypes])

# Add the constraints

# Solve the problem
model.maximize(objectiveFunction)
model.solve()
print(model.get_solve_status())
print(model.solution.get_objective_value())
print([float(value) for value in z.values()])

# Decision variables:
#   omega -- depends on S and lambda
#   v --
#   h -- dpends on lambda and k
#   y -- depends on m and s and lambda and i
#   z -- depends on S and lambda
#


# # Create a cplex model
# model = Model('BayesianPersuasionSolver')
# # Define the decision variables
# sW = mD.continuous_var(lb=float("-inf"),name='socialWelfare')
# xD = mD.continuous_var_dict(keys=dIds)
# # Constraints
# mD.add_constraint(sum(xD.values()) == 1)
# mD.add_constraints(xVal <= 1 for xVal in xD.values())
# mD.add_constraints(xVal >= 0 for xVal in xD.values())
# mD.add_constraints(vD <= sum([xD[dId] * payoutMatrix[dId, aId] for dId in dIds]) for aId in aIds)
#
# mD.maximize(vD)
# if export:
#     mD.export("defenderModel.lp")
# return mD, xD, vD

# def getDefenderMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export=False):
#     defenderModel, dStrategyDistribution, dUtility = createDefenderModel(dIds, dMap, aIds, aMap, payoutMatrix, export)
#     defenderModel.solve()
#     defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
#     dUtility = float(dUtility)
#     return defenderMixedStrategy, dUtility
