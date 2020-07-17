# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from docplex.mp.model import Model

#
# ONLY WORKS FOR 2 -- MAKE MORE ROBUST
#
def getOmegaKeys():
    # Place all defenders into the target slots
    # just use a list with target indices for each defender
    omegaKeys = []
    for placement in range(len(placements)):
        for aType in aTypes:
            omegaKeys.append((placement, aType))
    return tuple(omegaKeys)

def getPlacements():
    # Place all defenders into the target slots
    # just use a list with target indices for each defender
    placements = []
    for _ in range(targetNum**len(defenders)):
        placements.append((_ // targetNum,_ % targetNum))
    return placements

def assignmentProbability(lam, m, j):
    probability = 0
    for placement in placements:
        if placement[m] == j:
            probability += omega[lam, placement]
    return probability

def socialUtility():
    [assignmentProbability(lam,m,j) * dCosts[m][j] for j in range(targets) for m in defenders]


targetNum = 3
aTypes = list(range(2))
aRewards = {0:[1,7,3], 1:[2,1,5]}
aPenalties = {0:[2,2,2], 1:[1,1,2]}
q = [0.7, 0.3]
# Defenders
defenders = list(range(2))
dRewards = {0:[0,0,0],1:[0,0,0]}
dPenalties = {0:[1,2,3],1:[3,1,1]}
dCosts = {0:[1,1,2],1:[1,1,1]}
# Create the list of omega keys -- all possible assignments, with each defender
# type attached.
placements = getPlacements()
omegaKeys = getOmegaKeys()

model = Model('BayesianPersuasionSolver')
# omega = model.continuous_var_dict(keys=omegaKeys)
omega = dict.fromkeys(omegaKeys, 0.333)
objectiveFunction = 0
print(socialUtility())
model.maximize(objectiveFunction)


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
