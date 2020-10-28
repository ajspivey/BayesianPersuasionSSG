from docplex.mp.model import Model
from itertools import permutations
from functools import reduce
from operator import mul
from numpy import argmax
import numpy as np
from scipy import optimize
import networkx as nx
from time import time as getTime
import matplotlib.pyplot as plt
import random
import copy

from constants import DEFENDERNUM,ATTACKERNUM,TARGETNUM,AVGCOUNT,M,GAMEUTILITY,GAMEMODEL,GAMEEXTRAS
from util import generateRandomDefenders, generateRandomAttackers, numberToBase, \
                getPlacements, getOmegaKeys, defenderSocialUtility, utilityM, \
                aUtility, getLambdaPlacements, utilityDI, utilityLamI, \
                probabilityProtected, createGraph

defenderUtilities = []
solutionTimes = []
models = {}

# ==========
# GAME TYPES
# ==========
# ------------------------------------------------------------------------------
def solveBPAllowOverlap(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where defender assignments are allowed to overlap (more than one defender per target is allowed)"""
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
    return model.solution.get_objective_value(), model, None

# ------------------------------------------------------------------------------
def solveBPNoRequiredDefenderAssignment(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where defender assignments are allowed to overlap, with one dummy target (represents defenders not having to be assigned)"""
    _dRewards = copy.deepcopy(dRewards)
    _dPenalties = copy.deepcopy(dPenalties)
    _dCosts = copy.deepcopy(dCosts)
    for m in defenders:
        _dRewards[m].append(0)
        _dPenalties[m].append(0)
        _dCosts[m].append(0)
    placements = getPlacements(defenders, targetNum + 1)
    attackerActions = list(range(targetNum))
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)
    model = Model('BayesianPersuasionSolverWithoutRequiredAssignment')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
    objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a,defenders, _dRewards, _dCosts, _dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])

    model.add_constraints([sum([w[(s,a,lam)] * aUtility(s,a,lam, aPenalties, aRewards) for s in placements]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam, aPenalties, aRewards) for s in placements]) for a in attackerActions for b in attackerActions if a != b for lam in aTypes], names=[f"c att {lam} suggested {a}, but goes to {b}" for a in attackerActions for b in attackerActions if a != b for lam in aTypes])
    model.add_constraints([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) >= \
                           sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) \
                           for m in defenders for d in range(targetNum + 1) for e in range(targetNum + 1) if d != e for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model, None

# ------------------------------------------------------------------------------
def solveBPNOOD(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where defender assignments are not allowed to overlap, with one dummy target (represents one defender that does not have to be assigned)"""
    _dRewards = copy.deepcopy(dRewards)
    _dPenalties = copy.deepcopy(dPenalties)
    _dCosts = copy.deepcopy(dCosts)
    for m in defenders:
        _dRewards[m].append(0)
        _dPenalties[m].append(0)
        _dCosts[m].append(0)
    overlapPlacements = getPlacements(defenders, targetNum + 1)
    placements = list(filter(lambda x: len(set(x)) == len(x), overlapPlacements))
    attackerActions = list(range(targetNum))
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)
    model = Model('BayesianPersuasionSolverWithOverlap')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
    objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a,defenders, _dRewards, _dCosts, _dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] * aUtility(s,a,lam, aPenalties, aRewards) for s in placements]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam, aPenalties, aRewards) for s in placements]) for a in attackerActions for b in attackerActions if a != b for lam in aTypes], names=[f"c att {lam} suggested {a}, but goes to {b}" for a in attackerActions for b in attackerActions if a != b for lam in aTypes])
    model.add_constraints([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) >= \
                           sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) \
                           for m in defenders for d in range(targetNum + 1) for e in range(targetNum + 1) if d != e for lam in aTypes], names=[f"defender {m} suggested {d}, but goes to {e} with att {lam}" for m in defenders for d in range(targetNum + 1) for e in range(targetNum + 1) if d != e for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes], names=[f"sum must be 1 for att: {lam}" for lam in aTypes])
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model, None

# ------------------------------------------------------------------------------
def solveBPNOND(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where defender assignments are not allowed to overlap, with as many dummy targets as defenders (represents defenders not having to be assigned)"""
    _dRewards = copy.deepcopy(dRewards)
    _dPenalties = copy.deepcopy(dPenalties)
    _dCosts = copy.deepcopy(dCosts)
    for m in defenders:
        for defenderCount in defenders:
            _dRewards[m].append(0)
            _dPenalties[m].append(0)
            _dCosts[m].append(0)
    overlapPlacements = getPlacements(defenders, targetNum + len(defenders))
    placements = list(filter(lambda x: len(set(x)) == len(x), overlapPlacements))
    attackerActions = list(range(targetNum))
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)
    model = Model('BayesianPersuasionSolverWithOverlap')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
    objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a,defenders, _dRewards, _dCosts, _dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] * aUtility(s,a,lam, aPenalties, aRewards) for s in placements]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam, aPenalties, aRewards) for s in placements]) for a in attackerActions for b in attackerActions if a != b for lam in aTypes], names=[f"c att {lam} suggested {a}, but goes to {b}" for a in attackerActions for b in attackerActions if a != b for lam in aTypes])
    model.add_constraints([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) >= \
                           sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) \
                           for m in defenders for d in range(targetNum + len(defenders)) for e in range(targetNum + len(defenders)) if d != e for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model, None

def solveSmallBPNONDDualEllipsoid(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q, maxIterations=80):
    """A game where defender assignments are not allowed to overlap, with as many
    dummy targets as defenders (represents defenders not having to be assigned).
    This problem is the dual of the primal above, and is solved using the ellipsoid
    method."""
    totalTime = getTime()
    setupTime = getTime()
    _dRewards = copy.deepcopy(dRewards)
    _dPenalties = copy.deepcopy(dPenalties)
    _dCosts = copy.deepcopy(dCosts)
    _aRewards = copy.deepcopy(aRewards)
    _aPenalties = copy.deepcopy(aPenalties)
    for m in defenders:
        for defenderCount in defenders:
            _dRewards[m].append(0)
            _dPenalties[m].append(0)
            _dCosts[m].append(0)
        for lam in aTypes:
            _aRewards[lam].append(0)
            _aPenalties[lam].append(0)
    targetNumWithDummies = len(_dRewards[0])
    targetRange = list(range(targetNumWithDummies))
    # Get the placements that occur with no overlap
    overlapPlacements = getPlacements(defenders, targetNumWithDummies)
    placements = list(filter(lambda x: len(set(x)) == len(x), overlapPlacements))
    s = [(sd,sa) for sd in placements for sa in targetRange]
    # Generate the keys
    aKeys = [(t,tPrime,lam) for t in targetRange for tPrime in targetRange for lam in aTypes]
    bKeys = [(t,tPrime,d) for t in targetRange for tPrime in targetRange for d in defenders]
    # Get a random subset of the placements
    subsetCount = int(len(placements) * 0.001)
    if subsetCount == 0:
        subsetCount = len(placements) // 4
    subsetS = random.choices(s, k=subsetCount)
    # Solve the dual using column generation
    for _ in range(maxIterations):
        print(f"ITERATION: {_}")
        relaxedModel = Model('relaxedModel')
        g = relaxedModel.continuous_var_dict(keys=aTypes, lb=-1000, ub=1000, name="g") # unbounded
        a = relaxedModel.continuous_var_dict(keys=aKeys, lb=0, ub=1000, name="a") # No upper bound
        b = relaxedModel.continuous_var_dict(keys=bKeys, lb=0, ub=1000, name="b") # No upper bound
        objectiveFunction = sum([g[lam] for lam in aTypes])
        dualConstraints = relaxedModel.add_constraints([                            \
            sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,sa,lam,_aPenalties,_aRewards)) * a[sa,tPrime,lam] for tPrime in targetRange]) + \
            sum([(utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts)) * b[sd[d],tPrime,d]  for d in defenders for tPrime in targetRange]) + \
            g[lam] \
            >= q[lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties)  \
            for sd,sa in subsetS for lam in aTypes])
        relaxedModel.minimize(objectiveFunction)
        relaxedModel.solve() # Alpha and Beta have values for each instance of target and attacker
        relaxedModel.export(f"relaxedModel.lp") # Remove for time savings

        constraintValues = [                            \
            (
            sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,sa,lam,_aPenalties,_aRewards)) * float(a[sa,tPrime,lam]) for tPrime in targetRange]) + sum([(utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts)) * float(b[sd[d],tPrime,d])  for d in defenders for tPrime in targetRange]) + float(g[lam]),\
            q[lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties)\
            ) \
            for sd,sa in subsetS for lam in aTypes]
        epsilon = -1e-3

        violated = False
        for constraintIndex in range(len(dualConstraints)):
            lhs, rhs = constraintValues[constraintIndex]
            diff = lhs - rhs
            if diff < epsilon:
                violated = True
                print(f"constraint {constraintIndex + 1} violated")
                print(f"constraint has lhs: {lhs}")
                print(f"constraint has rhs: {rhs}")
                print(f"constraint has diff: {diff}")
        if violated:
            print("alpha")
            for k,v in a.items():
                if float(v) != 0:
                    print(k, float(v))
            print("beta")
            for k,v in b.items():
                if float(v) != 0:
                    print(k, float(v))
            print("gamma")
            for k,v in g.items():
                if float(v) != 0:
                    print(k, float(v))
            alreadyIn = True
            while alreadyIn:
                randomSubset = random.choices(s, k=1)[0]
                if randomSubset not in subsetS:
                    alreadyIn = False
            subsetS.append(randomSubset)
        else:
            print(f"\n\n Model value: {relaxedModel.solution.get_objective_value()}")
            return relaxedModel.solution.get_objective_value(), relaxedModel, relaxedModel.dual_values(dualConstraints)


def solveSmallBPNONDDualEllipsoid2(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q, maxIterations=500):
    """A game where defender assignments are not allowed to overlap, with as many
    dummy targets as defenders (represents defenders not having to be assigned).
    This problem is the dual of the primal above, and is solved using the ellipsoid
    method."""
    totalTime = getTime()
    setupTime = getTime()
    _dRewards = copy.deepcopy(dRewards)
    _dPenalties = copy.deepcopy(dPenalties)
    _dCosts = copy.deepcopy(dCosts)
    _aRewards = copy.deepcopy(aRewards)
    _aPenalties = copy.deepcopy(aPenalties)
    for m in defenders:
        for defenderCount in defenders:
            _dRewards[m].append(0)
            _dPenalties[m].append(0)
            _dCosts[m].append(0)
        for lam in aTypes:
            _aRewards[lam].append(0)
            _aPenalties[lam].append(0)
    targetNumWithDummies = len(_dRewards[0])
    targetRange = list(range(targetNumWithDummies))
    # Get the placements that occur with no overlap
    overlapPlacements = getPlacements(defenders, targetNumWithDummies)
    placements = list(filter(lambda x: len(set(x)) == len(x), overlapPlacements))
    s = [(sd,sa) for sd in placements for sa in targetRange]
    # Generate the keys
    aKeys = [(t,tPrime,lam) for t in targetRange for tPrime in targetRange for lam in aTypes]
    bKeys = [(t,tPrime,d) for t in targetRange for tPrime in targetRange for d in defenders]
    # Get a random subset of the placements
    subsetCount = int(len(placements) * 0.001)
    if subsetCount == 0:
        subsetCount = len(placements) // 4
    subsetS = random.choices(s, k=subsetCount)
    # Solve the dual using column generation
    for _ in range(maxIterations):
        print(f"ITERATION: {_}")
        relaxedModel = Model('relaxedModel')
        g = relaxedModel.continuous_var_dict(keys=aTypes, lb=-1000, ub=1000, name="g") # unbounded
        a = relaxedModel.continuous_var_dict(keys=aKeys, lb=0, ub=1000, name="a") # No upper bound
        b = relaxedModel.continuous_var_dict(keys=bKeys, lb=0, ub=1000, name="b") # No upper bound
        objectiveFunction = sum([g[lam] for lam in aTypes])
        dualConstraints = relaxedModel.add_constraints([                            \
            sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,sa,lam,_aPenalties,_aRewards)) * a[sa,tPrime,lam] for tPrime in targetRange]) + \
            sum([(utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts)) * b[sd[d],tPrime,d]  for d in defenders for tPrime in targetRange]) + \
            g[lam] \
            >= q[lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties)  \
            for sd,sa in subsetS for lam in aTypes])
        relaxedModel.minimize(objectiveFunction)
        relaxedModel.solve() # Alpha and Beta have values for each instance of target and attacker
        print(f"Solution value: {relaxedModel.solution.get_objective_value()}")
        alreadyIn = True
        while alreadyIn:
            randomSubset = random.choices(s, k=1)[0]
            if randomSubset not in subsetS:
                alreadyIn = False
        subsetS.append(randomSubset)

    return relaxedModel.solution.get_objective_value(), relaxedModel, relaxedModel.dual_values(dualConstraints)


def solveBPNONDDualEllipsoid(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q, maxIterations=500):
    """A game where defender assignments are not allowed to overlap, with as many
    dummy targets as defenders (represents defenders not having to be assigned).
    This problem is the dual of the primal above, and is solved using the ellipsoid
    method."""
    # Add extra dummy targets
    epsilon = -1e-3
    totalTime = getTime()
    setupTime = getTime()
    _dRewards = copy.deepcopy(dRewards)
    _dPenalties = copy.deepcopy(dPenalties)
    _dCosts = copy.deepcopy(dCosts)
    _aRewards = copy.deepcopy(aRewards)
    _aPenalties = copy.deepcopy(aPenalties)
    for m in defenders:
        for defenderCount in defenders:
            _dRewards[m].append(0)
            _dPenalties[m].append(0)
            _dCosts[m].append(0)
        for lam in aTypes:
            _aRewards[lam].append(0)
            _aPenalties[lam].append(0)
    targetNumWithDummies = len(_dRewards[0])
    targetRange = list(range(targetNumWithDummies))

    # Get the suggestions that occur with no overlap
    overlapPlacements = getPlacements(defenders, targetNumWithDummies)
    placements = list(filter(lambda x: len(set(x)) == len(x), overlapPlacements))
    s = [(sd,sa) for sd in placements for sa in targetRange]
    # Generate the keys
    aKeys = [(t,tPrime,lam) for t in targetRange for tPrime in targetRange for lam in aTypes]
    bKeys = [(t,tPrime,d) for t in targetRange for tPrime in targetRange for d in defenders]
    # Get a random subset of the placements
    subsetCount = int(len(placements) * 0.001)
    if subsetCount == 0:
        subsetCount = len(placements) // 4
    subsetS = random.choices(s, k=subsetCount)
    setupTime = getTime() - setupTime

    # Solve the dual using column generation
    modelingTime = 0
    totalSubProblemTime = 0
    subproblem1SetupTime = 0
    subproblem2SetupTime = 0
    graphMatchingTime = 0
    for _ in range(maxIterations):
        print(f"ITERATION: {_}")
        modelingStartTime = getTime()
        relaxedModel = Model('relaxedModel')
        g = relaxedModel.continuous_var_dict(keys=aTypes, lb=-1000, ub=1000, name="g") # unbounded
        a = relaxedModel.continuous_var_dict(keys=aKeys, lb=0, ub=1000, name="a") # No upper bound
        b = relaxedModel.continuous_var_dict(keys=bKeys, lb=0, ub=1000, name="b") # No upper bound

        objectiveFunction = sum([g[lam] for lam in aTypes])
        dualConstraints = relaxedModel.add_constraints([                            \
            sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,sa,lam,_aPenalties,_aRewards)) * a[sa,tPrime,lam] for tPrime in targetRange]) + \
            sum([(utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts)) * b[sd[d],tPrime,d]  for d in defenders for tPrime in targetRange]) + \
            g[lam] \
            >= q[lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties)  \
            for sd,sa in subsetS for lam in aTypes])
        relaxedModel.minimize(objectiveFunction)
        relaxedModel.solve() # Alpha and Beta have values for each instance of target and attacker
        print(relaxedModel.get_solve_status())
        print(f"Utility on iteration {_} = {relaxedModel.solution.get_objective_value()}")
        modelingTime += getTime() - modelingStartTime

        # For every lam,t0, split (9) into two subproblems and solve each.
        totalSubProblemTime = getTime()
        violatedConstraints = False
        signalsToBeAdded = []
        for lam in aTypes:
            for t0 in targetRange:
                # Subproblem 1
                subproblem1SetupTimeStart = getTime()
                # Build the weights
                weightMatrix = np.zeros(shape=(targetNumWithDummies,targetNumWithDummies))
                for d in defenders:
                    for t in targetRange:
                        if t != t0:
                            weightValue = -q[lam]*_dCosts[d][t] \
                                            + (_dRewards[d][t0] + _dCosts[d][t0] - _dPenalties[d][t0] - _dCosts[d][t]) * float(b[t, t0, d]) \
                                            + sum([(_dCosts[d][tPrime] - _dCosts[d][t]) * float(b[t,tPrime,d]) for tPrime in targetRange if tPrime != t0]) \
                                            + (_aPenalties[lam][t] - _aRewards[lam][t0]) * float(a[t0, t, lam])
                            weightMatrix[d][t] = weightValue
                        else:
                            weightValue = 100000
                            weightMatrix[d][t] = weightValue
                for d in range(len(defenders), targetNumWithDummies):
                    for t in targetRange:
                        weightValue = (_aRewards[lam][t] - _aRewards[lam][t0]) * float(a[t0,t,lam])
                        weightMatrix[d][t] = weightValue

                subproblem1SetupTime += getTime() - subproblem1SetupTimeStart
                # Solve the problem
                graphMatchingTimeStart = getTime()
                defenderIndexes, placementIndexes = optimize.linear_sum_assignment(weightMatrix)
                newPlacement = [defenderPlacement for defenderPlacement in placementIndexes[:len(defenders)]]
                newPlacementCost = weightMatrix[defenderIndexes, placementIndexes].sum()
                graphMatchingTime += getTime() - graphMatchingTimeStart

                # Check the value of this s using (9) -- if negative, add s to
                # the subset of solutions.
                value = sum([(aUtility(newPlacement,tPrime,lam,_aPenalties,_aRewards) - aUtility(newPlacement,t0,lam,_aPenalties,_aRewards)) * float(a[t0,tPrime,lam]) for tPrime in targetRange]) + \
                            sum([(utilityM(tPrime,newPlacement,t0,d,_dRewards,_dPenalties,_dCosts) - utilityM(newPlacement[d],newPlacement,t0,d,_dRewards,_dPenalties,_dCosts)) * float(b[newPlacement[d],tPrime,d]) for d in defenders for tPrime in targetRange]) - \
                            (q[lam] * defenderSocialUtility(newPlacement,t0,defenders,_dRewards,_dCosts,_dPenalties)) + float(g[lam])

                existsFromPreviousIteration = False
                if value < epsilon:
                    violatedConstraints = True
                    print("p1 conflict")
                    signal = (newPlacement,t0)
                    if signal not in subsetS and signal not in signalsToBeAdded:
                        signalsToBeAdded.append(signal)
                    elif signal not in signalsToBeAdded:
                        existsFromPreviousIteration = True

                # Subproblem 2
                # Fix each possible defender that coveres t0. For each of these, find
                # the best matching
                for d0 in defenders:
                    subproblem2SetupTimeStart = getTime()
                    edges = {}
                    # Set weights for the defenders
                    for d in defenders:
                        if d != d0:
                            edges[f"d_{d}"] = {}
                            for t in targetRange:
                                if t != t0:
                                    weightValue = (_aPenalties[lam][t] - _aPenalties[lam][t0]) * float(a[t0,t,lam]) + \
                                                    sum([(_dCosts[d][tPrime] - _dCosts[d][t] )* float(b[t,tPrime,d]) for tPrime in targetRange])  - \
                                                    q[lam] * _dCosts[d][t]
                                    edges[f"d_{d}"][f"t_{t}"] = {"weight": weightValue}
                    # Add buffer defenders to reach the target count
                    for d in range(len(defenders), targetNumWithDummies):
                        edges[f"ed_{d}"] = {}
                        for t in targetRange:
                            if t != t0:
                                weightValue = (_aRewards[lam][t] - _aPenalties[lam][t0]) * float(a[t0,t,lam])
                                edges[f"ed_{d}"][f"t_{t}"] = {"weight": weightValue}
                    subproblem2SetupTime += getTime() - subproblem2SetupTimeStart
                    # Build the graph
                    graphMatchingTimeStart = getTime()
                    G = nx.from_dict_of_dicts(edges)
                    # Solve the problem
                    matchings = nx.algorithms.bipartite.minimum_weight_full_matching(G)

                    # Convert the solution back into our setting
                    newPlacement = [0] * len(defenders)
                    for k,v in matchings.items():
                        if k.startswith("d_"):
                            defender = int(k.split("_")[1])
                            target = int(v.split("_")[1])
                            newPlacement[defender] = target
                    newPlacement[d0] = t0
                    graphMatchingTime += getTime() - graphMatchingTimeStart

                    # Check the value of this s using (9) -- if negative, add s to
                    # the subset of solutions.
                    value = sum([(aUtility(newPlacement,tPrime,lam,_aPenalties,_aRewards) - aUtility(newPlacement,t0,lam,_aPenalties,_aRewards)) * float(a[t0,tPrime,lam]) for tPrime in targetRange]) + \
                                sum([(utilityM(tPrime,newPlacement,t0,d,_dRewards,_dPenalties,_dCosts) - utilityM(newPlacement[d],newPlacement,t0,d,_dRewards,_dPenalties,_dCosts)) * float(b[newPlacement[d],tPrime,d]) for d in defenders for tPrime in targetRange]) - \
                                (q[lam] * defenderSocialUtility(newPlacement,t0,defenders,_dRewards,_dCosts,_dPenalties))
                    if value < epsilon:
                        print(value)
                        violatedConstraints = True
                        signal = (newPlacement,t0)
                        print(f"signal {signal}")
                        print(signal in subsetS)
                        if signal not in subsetS and signal not in signalsToBeAdded:
                            signalsToBeAdded.append(signal)

        # Now that we have checked all the violated constraints, either return
        # the solution ( get the dual values) or recompute the optimal value of
        # the dual with additional constraints
        for signal in signalsToBeAdded:
            subsetS.append(signal)
        if not violatedConstraints:
            print("NO VIOLATED CONSTRAINTS")
            totalSubProblemTime = getTime() - totalSubProblemTime
            totalTime = getTime() - totalTime
            print(f"Total Time: {totalTime}")
            print(f"Setup Time: {setupTime}")
            print(f"Modeling Time: {modelingTime}")
            print(f"Total Sub Problem Time: {totalSubProblemTime}")
            print(f"Sub Problem 1 setup Time: {subproblem1SetupTime}")
            print(f"Sub Problem 2 setup Time: {subproblem2SetupTime}")
            print(f"Graph Matching Time: {graphMatchingTime}")
            print(f"Total Time: {totalTime}")
            print(f"Iteration Count: {_}")
            print(relaxedModel.dual_values(dualConstraints))
            return relaxedModel.solution.get_objective_value(), relaxedModel, relaxedModel.dual_values(dualConstraints)
    totalSubProblemTimeStart = getTime() - totalSubProblemTime
    totalTime = getTime() - totalTime
    print(f"Total Time: {totalTime}")
    print(f"Setup Time: {setupTime}")
    print(f"Modeling Time: {modelingTime}")
    print(f"Total Sub Problem Time: {totalSubProblemTime}")
    print(f"Sub Problem 1 Time: {subproblem1SetupTime}")
    print(f"Sub Problem 2 Time: {subproblem2SetupTime}")
    print(f"Graph Matching Time: {graphMatchingTime}")
    print(f"Total Time: {totalTime}")
    print(f"Iteration Count: 100 -- max iterations")
    print(relaxedModel.dual_values(dualConstraints))
    return relaxedModel.solution.get_objective_value(), relaxedModel, relaxedModel.dual_values(dualConstraints)


# ------------------------------------------------------------------------------
def solveBaseline(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where self-interested defenders optimize in a bayesian setting and the attacker performs a best-response to their strategy"""
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
    return baselineUtility, models, None
