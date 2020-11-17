from docplex.mp.model import Model
from itertools import permutations
from functools import reduce
from operator import mul
from numpy import argmax
import numpy as np
import networkx as nx
from time import time as getTime
import matplotlib.pyplot as plt
import random
import copy

from constants import DEFENDERNUM,ATTACKERNUM,TARGETNUM,AVGCOUNT,M,GAMEUTILITY,GAMEMODEL,GAMEEXTRAS,EPSILON
from util import generateRandomDefenders, generateRandomAttackers, numberToBase, \
                getPlacements, getOmegaKeys, defenderSocialUtility, utilityM, \
                aUtility, getLambdaPlacements, utilityDI, utilityLamI, \
                probabilityProtected, createGraph

defenderUtilities = []
solutionTimes = []
models = {}

# ==============================================================================
# GAME TYPES
# ==============================================================================

# =======
# EX POST
# =======
# ------------------------------------------------------------------------------
def solveBaseline(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where self-interested defenders optimize in a bayesian setting and
    the attacker performs a best-response to their strategy"""
    """Contains a dummy target for defenders and attackers"""
    # Add the dummy target
    _dRewards = copy.deepcopy(dRewards)
    _dPenalties = copy.deepcopy(dPenalties)
    _dCosts = copy.deepcopy(dCosts)
    _aRewards = copy.deepcopy(aRewards)
    _aPenalties = copy.deepcopy(aPenalties)
    for m in defenders:
        _dRewards[m].append(0)
        _dPenalties[m].append(0)
        _dCosts[m].append(0)
    for lam in aTypes:
        _aRewards[lam].append(0)
        _aPenalties[lam].append(0)
    targetNumWithDummies = len(_dRewards[0])
    targetRange = list(range(targetNumWithDummies))

    # Generate the placements and keys
    placements = getPlacements(defenders, targetNumWithDummies)
    attackerActions = targetRange
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)

    # Construct the model for each defender
    baselineUtility = 0
    dStrat = {}
    models2 = {}
    for m in defenders:
        model2 = Model(f"defenderStrategy{m}")
        x = model2.continuous_var_list(keys=targetNumWithDummies, lb=0, ub=1, name=f"x{m}")
        h = model2.binary_var_dict(keys=[(lam, k) for lam in aTypes for k in targetRange], lb=0, ub=1, name=f"h{m}")
        ul = model2.continuous_var_dict(keys=aTypes, lb=-model2.infinity, name=f"ua{m}")
        ud = model2.continuous_var_dict(keys=[lam for lam in aTypes], lb=-model2.infinity, name=f"ud{m}")
        objectiveFunction = sum([q[lam] * ud[lam] for lam in aTypes])
        model2.add_constraints([ud[lam] <= utilityDI(m,x,i,_dRewards,_dPenalties,_dCosts) + (1-h[(lam,i)]) * M for i in targetRange for lam in aTypes], names=[f"defender utility for lam {lam}, i {i}" for i in targetRange for lam in aTypes])
        model2.add_constraints([ul[lam] <= utilityLamI(x,lam,i,_aRewards,_aPenalties) + (1-h[(lam,i)]) * M for i in targetRange for lam in aTypes], names=[f"lam {lam} utility leq for i {i}" for i in targetRange for lam in aTypes])
        model2.add_constraints([ul[lam] >= utilityLamI(x,lam,i,_aRewards,_aPenalties) for i in targetRange for lam in aTypes], names=[f"lam {lam} utility geq, for i {i}" for i in targetRange for lam in aTypes])
        model2.add_constraints([sum([h[(lam,i)] for i in targetRange]) == 1 for lam in aTypes], names=[f"h sum is 1 for lam {lam}" for lam in aTypes])
        model2.add_constraint(sum([x[i] for i in targetRange]) == 1)
        # Solve the model for each defender
        model2.maximize(objectiveFunction)
        model2.solve()
        model2.export("baselineModel.lp")
        dStrat[m] = list([float(xVal) for xVal in x])
        models[m] = model2
    # Attacker best response (for each attacker type)
    aStrat = {}
    protectionOdds = probabilityProtected(dStrat, targetNumWithDummies)
    for lam in aTypes:
        expectedUtilities = []
        for i in targetRange:
            expectedUtilities.append(((1-protectionOdds[i])*_aRewards[lam][i]) + (protectionOdds[i]*_aPenalties[lam][i]))
        aStrat[lam] = argmax(expectedUtilities)
    # Calculate defender expected utility for attacker best response
    for m in defenders:
        for lam in aTypes:
            attackedTarget = aStrat[lam]                                                                                        # The target attacked by this attacker
            coveredUtility = protectionOdds[attackedTarget] * (_dRewards[m][attackedTarget])                                    # The expected utility we catch this attacker
            uncoveredUtility = (1-protectionOdds[attackedTarget]) * (_dPenalties[m][attackedTarget])                            # The expected utility we miss this attacker
            baselineUtility +=  q[lam] * (coveredUtility + uncoveredUtility)
        expectedCost = sum([dStrat[m][target] * _dCosts[m][target] for target in targetRange])
        baselineUtility += expectedCost
    return baselineUtility, models, None

# ------------------------------------------------------------------------------
def solvePrimalOverlap(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where defender assignments are allowed to overlap"""
    """Contains a dummy target for defenders and attackers"""
    # Add the extra dummy target
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
    attackerActions = targetRange
    placements = getPlacements(defenders, targetNumWithDummies)
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)

    # Build the model
    model = Model('PrimalWithOverlap')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
    objectiveFunction = sum([q[lam] * sum([w[s,a,lam] * defenderSocialUtility(s,a,defenders,_dRewards,_dCosts,_dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])
    c1 = [sum([w[s,a,lam] * aUtility(s,a,lam,_aPenalties,_aRewards) for s in placements]) \
                            >= sum([w[s,a,lam] * aUtility(s,b,lam,_aPenalties,_aRewards) for s in placements])
                            for lam in aTypes for a in attackerActions for b in attackerActions if a != b]
    c1 = [constraint for constraint in c1 if not isinstance(constraint, bool)]
    c1 = model.add_constraints(c1)
    c2 = model.add_constraints([sum([q[lam] * sum([w[s,a,lam] * utilityM(d,s,a,m,_dRewards,_dPenalties,_dCosts) for a in attackerActions for s in placements if s[m] == d]) for lam in aTypes]) \
                            >= sum([q[lam] * sum([w[s,a,lam] * utilityM(e,s,a,m,_dRewards,_dPenalties,_dCosts) for a in attackerActions for s in placements if s[m] == d]) for lam in aTypes])
                            for m in defenders for d in targetRange for e in targetRange if d!=e])
    c3 = model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
    # Solve the model
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model, None

# ------------------------------------------------------------------------------
def solvePrimalNoOverlap(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where defender assignments are not allowed to overlap"""
    """Contains as many dummy targets as defenders, for defenders and attackers"""
    # Add the extra dummy targets
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
    attackerActions = targetRange
    # Get the suggestions that occur with no overlap
    overlapPlacements = getPlacements(defenders, targetNumWithDummies)
    placements = list(filter(lambda x: len(set(x)) == len(x), overlapPlacements))
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)

    # Build the model
    model = Model('PrimalWithOverlap')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
    objectiveFunction = sum([q[lam] * sum([w[s,a,lam] * defenderSocialUtility(s,a,defenders,_dRewards,_dCosts,_dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])
    c1 = [sum([w[s,a,lam] * aUtility(s,a,lam,_aPenalties,_aRewards) for s in placements]) \
                            >= sum([w[s,a,lam] * aUtility(s,b,lam,_aPenalties,_aRewards) for s in placements])
                            for lam in aTypes for a in attackerActions for b in attackerActions if a != b]
    c1 = [constraint for constraint in c1 if not isinstance(constraint, bool)]
    c1 = model.add_constraints(c1)
    c2 = model.add_constraints([sum([q[lam] * sum([w[s,a,lam] * utilityM(d,s,a,m,_dRewards,_dPenalties,_dCosts) for a in attackerActions for s in placements if s[m] == d]) for lam in aTypes]) \
                            >= sum([q[lam] * sum([w[s,a,lam] * utilityM(e,s,a,m,_dRewards,_dPenalties,_dCosts) for a in attackerActions for s in placements if s[m] == d]) for lam in aTypes])
                            for m in defenders for d in targetRange for e in targetRange if d!=e])
    c3 = model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
    # Solve the model
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model, None


def solveDualEllipsoid(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q, maxIterations=500):
    """A game where defender assignments are not allowed to overlap"""
    """Contains as many dummy targets as defenders, for defenders and attackers"""
    """This problem is the dual of the primal above, and is solved using the ellipsoid method."""
    # Add extra dummy targets
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

    # Create the model
    relaxedModel = Model('relaxedModel')
    # Create the variables
    g = relaxedModel.continuous_var_dict(keys=aTypes, lb=-1000, ub=1000, name="g") # unbounded
    a = relaxedModel.continuous_var_dict(keys=aKeys, lb=0, ub=1000, name="a") # No upper bound
    b = relaxedModel.continuous_var_dict(keys=bKeys, lb=0, ub=1000, name="b") # No upper bound
    # objective function:
    objectiveFunction = sum([g[lam] for lam in aTypes])
    # Initial constraints
    dualConstraints = relaxedModel.add_constraints([                            \
        sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,sa,lam,_aPenalties,_aRewards)) * a[sa,tPrime,lam] for tPrime in targetRange]) + \
        q[lam] * sum([(utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts)) * b[sd[d],tPrime,d]  for d in defenders for tPrime in targetRange]) + \
        g[lam] \
        >= q[lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties)  \
        for sd,sa in subsetS for lam in aTypes])

    # Solve the dual using column generation
    for _ in range(maxIterations):
        # print(f"ITERATION: {_}")

        relaxedModel.minimize(objectiveFunction)
        relaxedModel.solve() # Alpha and Beta have values for each instance of target and attacker
        # print(f"Utility on iteration {_} = {relaxedModel.solution.get_objective_value()}")

        # For every lam,t0, split (9) into two subproblems and solve each.
        violatedConstraints = False
        signalsToBeAdded = []
        for lam in aTypes:
            for t0 in targetRange:
                # Subproblem 1
                edges = {}
                # Graph weights with normal defenders
                for d in defenders:
                    edges[f"d_{d}"] = {}
                    for t in targetRange:
                        if t == t0:
                            weightValue = 10000000
                        else:
                            weightValue = (-q[lam]) * _dCosts[d][t] \
                                        + q[lam] * (_dRewards[d][t0] + _dCosts[d][t0] - _dPenalties[d][t0] - _dCosts[d][t]) * float(b[t,t0,d]) \
                                        + q[lam] * (sum([(_dCosts[d][tPrime] - _dCosts[d][t]) * float(b[t,tPrime,d]) for tPrime in targetRange if tPrime != t0])) \
                                        + (_aPenalties[lam][t] - _aRewards[lam][t0]) * float(a[t0,t,lam])
                        edges[f"d_{d}"][f"t_{t}"] = {"weight": weightValue}
                # Graph weights with added defenders
                for d in range(len(defenders), targetNumWithDummies):
                    edges[f"ed_{d}"] = {}
                    for t in targetRange:
                        weightValue = (_aRewards[lam][t] - _aRewards[lam][t0]) * float(a[t0,t,lam])
                        edges[f"ed_{d}"][f"t_{t}"] = {"weight": weightValue}

                # Solve the problem
                G = nx.from_dict_of_dicts(edges)
                matchings = nx.algorithms.bipartite.minimum_weight_full_matching(G)
                newPlacement = [0] * len(defenders)
                for k,v in matchings.items():
                    if k.startswith("d_"):
                        defender = int(k.split("_")[1])
                        target = int(v.split("_")[1])
                        newPlacement[defender] = target

                # Check the value of this s using (10) -- if negative, add s to
                # the subset of solutions.
                value = sum([(aUtility(newPlacement,tPrime,lam,_aPenalties,_aRewards) - aUtility(newPlacement,t0,lam,_aPenalties,_aRewards)) * float(a[t0,tPrime,lam]) for tPrime in targetRange])\
                            + q[lam] * sum([(utilityM(tPrime,newPlacement,t0,d,_dRewards,_dPenalties,_dCosts) - utilityM(newPlacement[d],newPlacement,t0,d,_dRewards,_dPenalties,_dCosts)) * float(b[newPlacement[d],tPrime,d]) for d in defenders for tPrime in targetRange])\
                            - (q[lam] * defenderSocialUtility(newPlacement,t0,defenders,_dRewards,_dCosts,_dPenalties)) + float(g[lam])

                if value < EPSILON:
                    signal = (newPlacement,t0)
                    if signal not in signalsToBeAdded:
                        violatedConstraints = True
                        signalsToBeAdded.append(signal)

                # Subproblem 2
                # Fix each possible defender that coveres t0. For each of these, find
                # the best matching
                for d0 in defenders:
                    edges = {}
                    # Graph weights with normal defenders (minus d0 and t0)
                    for d in defenders:
                        if d != d0:
                            edges[f"d_{d}"] = {}
                            for t in targetRange:
                                if t != t0:
                                    weightValue = (_aPenalties[lam][t] - _aPenalties[lam][t0]) * float(a[t0,t,lam]) \
                                                + q[lam] * (sum([(_dCosts[d][tPrime] - _dCosts[d][t]) * float(b[t,tPrime,d]) for tPrime in targetRange])) \
                                                - (q[lam] * _dCosts[d][t])
                                    edges[f"d_{d}"][f"t_{t}"] = {"weight": weightValue}
                    # Graph weights with added defenders (minus t0)
                    for d in range(len(defenders), targetNumWithDummies):
                        edges[f"ed_{d}"] = {}
                        for t in targetRange:
                            if t != t0:
                                weightValue = (_aRewards[lam][t] - _aPenalties[lam][t0]) * float(a[t0,t,lam])
                                edges[f"ed_{d}"][f"t_{t}"] = {"weight": weightValue}

                    # Solve the problem
                    G = nx.from_dict_of_dicts(edges)
                    matchings = nx.algorithms.bipartite.minimum_weight_full_matching(G)
                    newPlacement = [0] * len(defenders)
                    for k,v in matchings.items():
                        if k.startswith("d_"):
                            defender = int(k.split("_")[1])
                            target = int(v.split("_")[1])
                            newPlacement[defender] = target
                    newPlacement[d0] = t0

                    # Check the value of this s using (9) -- if negative, add s to
                    # the subset of solutions.
                    value = sum([(aUtility(newPlacement,tPrime,lam,_aPenalties,_aRewards) - aUtility(newPlacement,t0,lam,_aPenalties,_aRewards)) * float(a[t0,tPrime,lam]) for tPrime in targetRange])\
                                + q[lam] * sum([(utilityM(tPrime,newPlacement,t0,d,_dRewards,_dPenalties,_dCosts) - utilityM(newPlacement[d],newPlacement,t0,d,_dRewards,_dPenalties,_dCosts)) * float(b[newPlacement[d],tPrime,d]) for d in defenders for tPrime in targetRange])\
                                - (q[lam] * defenderSocialUtility(newPlacement,t0,defenders,_dRewards,_dCosts,_dPenalties)) + float(g[lam])

                    if value < EPSILON:
                        signal = (newPlacement,t0)
                        if signal not in signalsToBeAdded:
                            violatedConstraints = True
                            signalsToBeAdded.append(signal)

        # Now that we have checked all the violated constraints, either return
        # the solution ( get the dual values) or recompute the optimal value of
        # the dual with additional constraints
        newConstraints = relaxedModel.add_constraints([                            \
            sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,sa,lam,_aPenalties,_aRewards)) * a[sa,tPrime,lam] for tPrime in targetRange]) + \
            q[lam] * sum([(utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts)) * b[sd[d],tPrime,d]  for d in defenders for tPrime in targetRange]) + \
            g[lam] \
            >= q[lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties)  \
            for sd,sa in signalsToBeAdded for lam in aTypes])
        for signal in signalsToBeAdded:
            subsetS.append(signal)
        if not violatedConstraints:
            # print(relaxedModel.dual_values(relaxedModel.iter_constraints()))
            return relaxedModel.solution.get_objective_value(), relaxedModel, None#relaxedModel.dual_values(relaxedModel.iter_constraints())
    # print(relaxedModel.dual_values(relaxedModel.iter_constraints()))
    return relaxedModel.solution.get_objective_value(), relaxedModel, None#relaxedModel.dual_values(relaxedModel.iter_constraints())

# =======
# EX ANTE
# =======
# ------------------------------------------------------------------------------
def solvePrimalOverlapEX(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where defender assignments are allowed to overlap"""
    """Contains a dummy target for defenders and attackers"""
    """In this game the attacker and defender reason ex-ante
    (they choose to follow signals or not before a signal is sent)."""
    # Add the extra dummy target
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
    attackerActions = targetRange
    placements = getPlacements(defenders, targetNumWithDummies)
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)

    # Build the model
    model = Model('ExAnteWithOverlap')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
    objectiveFunction = sum([q[lam] * sum([w[sd,sa,lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties) for sd in placements for sa in attackerActions]) for lam in aTypes])
    # Define the constraints
    c1 = [sum([w[sd,sa,lam] * aUtility(sd,sa,lam,_aPenalties,_aRewards) for sd in placements for sa in attackerActions]) \
        >= sum([w[sd,sa,lam] * aUtility(sd,tPrime,lam,_aPenalties,_aRewards) for sd in placements for sa in attackerActions])
        for lam in aTypes for tPrime in targetRange]
    c1 = [constraint for constraint in c1 if not isinstance(constraint, bool)]
    c2 = [sum([q[lam] * sum([w[sd,sa,lam] * utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts) for sd in placements for sa in attackerActions]) for lam in aTypes]) \
        >= sum([q[lam] * sum([w[sd,sa,lam] * utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) for sd in placements for sa in attackerActions]) for lam in aTypes]) \
        for d in defenders for tPrime in targetRange]
    c3 = [sum([w[sd,sa,lam] for sd in placements for sa in attackerActions]) == 1
        for lam in aTypes]
    # Add the constraints
    c1 = model.add_constraints(c1)
    c2 = model.add_constraints(c2)
    c3 = model.add_constraints(c3)
    # Solve the model
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model, None

# ------------------------------------------------------------------------------
def solvePrimalNoOverlapEX(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where defender assignments are not allowed to overlap"""
    """Contains as many dummy targets as defenders, for defenders and attackers"""
    """In this game the attacker and defender reason ex-ante
    (they choose to follow signals or not before a signal is sent)."""
    # Add the extra dummy targets
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
    attackerActions = targetRange
    # Get the suggestions that occur with no overlap
    overlapPlacements = getPlacements(defenders, targetNumWithDummies)
    placements = list(filter(lambda x: len(set(x)) == len(x), overlapPlacements))
    omegaKeys = getOmegaKeys(aTypes, placements, attackerActions)

    # Build the model
    model = Model('ExAnteWithOverlap')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, ub=1, name="w")
    objectiveFunction = sum([q[lam] * sum([w[sd,sa,lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties) for sd in placements for sa in attackerActions]) for lam in aTypes])
    # Define the constraints
    c1 = [sum([w[sd,sa,lam] * aUtility(sd,sa,lam,_aPenalties,_aRewards) for sd in placements for sa in attackerActions]) \
        >= sum([w[sd,sa,lam] * aUtility(sd,tPrime,lam,_aPenalties,_aRewards) for sd in placements for sa in attackerActions])
        for lam in aTypes for tPrime in targetRange]
    c1 = [constraint for constraint in c1 if not isinstance(constraint, bool)]
    c2 = [sum([q[lam] * sum([w[sd,sa,lam] * utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts) for sd in placements for sa in attackerActions]) for lam in aTypes]) \
        >= sum([q[lam] * sum([w[sd,sa,lam] * utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) for sd in placements for sa in attackerActions]) for lam in aTypes]) \
        for d in defenders for tPrime in targetRange]
    c3 = [sum([w[sd,sa,lam] for sd in placements for sa in attackerActions]) == 1
        for lam in aTypes]
    # Add the constraints
    c1 = model.add_constraints(c1)
    c2 = model.add_constraints(c2)
    c3 = model.add_constraints(c3)
    # Solve the model
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model, None


def solveDualEllipsoidEX(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q, maxIterations=500):
    """A game where defender assignments are not allowed to overlap"""
    """Contains as many dummy targets as defenders, for defenders and attackers"""
    """This problem is the dual of the primal above, and is solved using the ellipsoid method."""
    # Add extra dummy targets
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
    aKeys = [(t,lam) for t in targetRange for lam in aTypes]
    bKeys = [(t,d) for t in targetRange for d in defenders]

    # Get a random subset of the placements
    subsetCount = int(len(placements) * 0.001)
    if subsetCount == 0:
        subsetCount = len(placements) // 4
    subsetS = random.choices(s, k=subsetCount)

    # Create the model
    relaxedModel = Model('relaxedModel')
    # Create the variables
    g = relaxedModel.continuous_var_dict(keys=aTypes, lb=-1000, ub=1000, name="g") # unbounded
    a = relaxedModel.continuous_var_dict(keys=aKeys, lb=0, ub=1000, name="a") # No upper bound
    b = relaxedModel.continuous_var_dict(keys=bKeys, lb=0, ub=1000, name="b") # No upper bound
    # objective function:
    objectiveFunction = sum([g[lam] for lam in aTypes])
    # Initial constraints

    ##
    ## CHANGED
    ##
    dualConstraints = relaxedModel.add_constraints([\
        sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,sa,lam,_aPenalties,_aRewards)) * a[tPrime,lam] for tPrime in targetRange]) \
        + sum([q[lam] * (utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts)) * b[tPrime,d] for d in defenders for tPrime in targetRange]) \
        + g[lam] \
        >= q[lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties)
    for sd,sa in subsetS for lam in aTypes])

    # Solve the dual using column generation
    for _ in range(maxIterations):
        relaxedModel.minimize(objectiveFunction)
        relaxedModel.solve() # Alpha and Beta have values for each instance of target and attacker

        # For every lam,t0, split (9) into two subproblems and solve each.
        violatedConstraints = False
        signalsToBeAdded = []
        for lam in aTypes:
            for t0 in targetRange:
                # Subproblem 1
                edges = {}
                # Graph weights with normal defenders
                for d in defenders:
                    edges[f"d_{d}"] = {}
                    for t in targetRange:
                        if t == t0:
                            weightValue = 10000000
                        else:
                            ##
                            ## CHANGED
                            ##
                            weightValue = ((_aPenalties[lam][t] - _aRewards[lam][t0]) * float(a[t,lam])) \
                                        + (q[lam] * sum([(_dCosts[d][tPrime] - _dCosts[d][t]) * float(b[tPrime,d]) for tPrime in targetRange if tPrime != t0])) \
                                        + (q[lam] * sum([(_dCosts[d][t0] + _dRewards[d][t0] - _dCosts[d][t] - _dPenalties[d][t0]) * float(b[t0,d]) for d in defenders])) \
                                        - (q[lam] * _dCosts[d][t])
                        edges[f"d_{d}"][f"t_{t}"] = {"weight": weightValue}
                # Graph weights with added defenders
                for d in range(len(defenders), targetNumWithDummies):
                    edges[f"ed_{d}"] = {}
                    for t in targetRange:
                        ##
                        ## CHANGED
                        ##
                        weightValue = (_aRewards[lam][t] - _aRewards[lam][t0]) * float(a[t,lam])
                        edges[f"ed_{d}"][f"t_{t}"] = {"weight": weightValue}

                # Solve the problem
                G = nx.from_dict_of_dicts(edges)
                matchings = nx.algorithms.bipartite.minimum_weight_full_matching(G)
                newPlacement = [0] * len(defenders)
                for k,v in matchings.items():
                    if k.startswith("d_"):
                        defender = int(k.split("_")[1])
                        target = int(v.split("_")[1])
                        newPlacement[defender] = target

                # Check the value of this s using (10) -- if negative, add s to
                # the subset of solutions.
                ##
                ## CHANGED
                ##
                value = sum([(aUtility(newPlacement,tPrime,lam,_aPenalties,_aRewards) - aUtility(newPlacement,t0,lam,_aPenalties,_aRewards)) * float(a[tPrime,lam]) for tPrime in targetRange]) \
                    + sum([q[lam] * (utilityM(tPrime,newPlacement,t0,d,_dRewards,_dPenalties,_dCosts) - utilityM(newPlacement[d],newPlacement,t0,d,_dRewards,_dPenalties,_dCosts)) * float(b[tPrime,d]) for d in defenders for tPrime in targetRange]) \
                    - (q[lam] * defenderSocialUtility(newPlacement,t0,defenders,_dRewards,_dCosts,_dPenalties)) + float(g[lam])

                if value < EPSILON:
                    signal = (newPlacement,t0)
                    if signal not in signalsToBeAdded:
                        violatedConstraints = True
                        signalsToBeAdded.append(signal)

                # Subproblem 2
                # Fix each possible defender that coveres t0. For each of these, find
                # the best matching
                for d0 in defenders:
                    edges = {}
                    # Graph weights with normal defenders (minus d0 and t0)
                    for d in defenders:
                        if d != d0:
                            edges[f"d_{d}"] = {}
                            for t in targetRange:
                                if t != t0:
                                    ##
                                    ## CHANGE
                                    ##
                                    weightValue = (_aPenalties[lam][t] - _aPenalties[lam][t0]) * float(a[t,lam]) \
                                                + q[lam] * (sum([(_dCosts[d][tPrime] - _dCosts[d][t]) * float(b[tPrime,d]) for tPrime in targetRange])) \
                                                - (q[lam] * _dCosts[d][t])
                                    edges[f"d_{d}"][f"t_{t}"] = {"weight": weightValue}
                    # Graph weights with added defenders (minus t0)
                    for d in range(len(defenders), targetNumWithDummies):
                        edges[f"ed_{d}"] = {}
                        for t in targetRange:
                            if t != t0:
                                ##
                                ## CHANGE
                                ##
                                weightValue = (_aRewards[lam][t] - _aPenalties[lam][t0]) * float(a[t,lam])
                                edges[f"ed_{d}"][f"t_{t}"] = {"weight": weightValue}

                    # Solve the problem
                    G = nx.from_dict_of_dicts(edges)
                    matchings = nx.algorithms.bipartite.minimum_weight_full_matching(G)
                    newPlacement = [0] * len(defenders)
                    for k,v in matchings.items():
                        if k.startswith("d_"):
                            defender = int(k.split("_")[1])
                            target = int(v.split("_")[1])
                            newPlacement[defender] = target
                    newPlacement[d0] = t0

                    # Check the value of this s using (9) -- if negative, add s to
                    # the subset of solutions.
                    ##
                    ## CHANGED
                    ##
                    value = sum([(aUtility(newPlacement,tPrime,lam,_aPenalties,_aRewards) - aUtility(newPlacement,t0,lam,_aPenalties,_aRewards)) * float(a[tPrime,lam]) for tPrime in targetRange]) \
                        + sum([q[lam] * (utilityM(tPrime,newPlacement,t0,d,_dRewards,_dPenalties,_dCosts) - utilityM(newPlacement[d],newPlacement,t0,d,_dRewards,_dPenalties,_dCosts)) * float(b[tPrime,d]) for d in defenders for tPrime in targetRange]) \
                        - (q[lam] * defenderSocialUtility(newPlacement,t0,defenders,_dRewards,_dCosts,_dPenalties)) + float(g[lam])

                    if value < EPSILON:
                        signal = (newPlacement,t0)
                        if signal not in signalsToBeAdded:
                            violatedConstraints = True
                            signalsToBeAdded.append(signal)

        # Now that we have checked all the violated constraints, either return
        # the solution ( get the dual values) or recompute the optimal value of
        # the dual with additional constraints
        ##
        ## CHANGED
        ##
        newConstraints = relaxedModel.add_constraints([\
            sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,sa,lam,_aPenalties,_aRewards)) * a[tPrime,lam] for tPrime in targetRange]) \
            + sum([q[lam] * (utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts)) * b[tPrime,d] for d in defenders for tPrime in targetRange]) \
            + g[lam] \
            >= q[lam] * defenderSocialUtility(sd,sa,defenders,_dRewards,_dCosts,_dPenalties)
        for sd,sa in signalsToBeAdded for lam in aTypes])
        for signal in signalsToBeAdded:
            subsetS.append(signal)
        if not violatedConstraints:
            return relaxedModel.solution.get_objective_value(), relaxedModel, None#relaxedModel.dual_values(relaxedModel.iter_constraints())
    return relaxedModel.solution.get_objective_value(), relaxedModel, None#relaxedModel.dual_values(relaxedModel.iter_constraints())

# ------------------------------------------------------------------------------
def solveCompactEX(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A compact representation of the EX gamse"""
    """In this game the attacker and defender reason ex-ante
    (they choose to follow signals or not before a signal is sent)."""
    # Add the extra dummy targets
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
    attackerActions = targetRange
    # Get the suggestions that occur with no overlap
    omegaKeys = [(t,d,tPrime,lam) for t in targetRange for d in defenders for tPrime in targetRange for lam in aTypes]
    omegaKeys2 = [(t,lam) for t in targetRange for lam in aTypes]

    # Build the model
    model = Model('ExAnteWithOverlap')
    w = model.continuous_var_dict(keys=omegaKeys, lb=0, name="w")
    w2 = model.continuous_var_dict(keys=omegaKeys2, lb=0, name="w2")
    objectiveFunction = sum([q[lam] * sum([w[t,d,t,lam] * _dRewards[d][t] for t in targetRange for d in defenders]) for lam in aTypes]) \
                        + sum([sum([(w2[t,lam] - sum([w[t,d,t,lam] for d in defenders])) * sum([_dPenalties[d][t] for d in defenders]) for t in targetRange]) for lam in aTypes]) \
                        + sum([q[lam] * sum([sum([w[t,d,tPrime,lam] for t in targetRange]) * _dCosts[d][tPrime] for tPrime in targetRange for d in defenders]) for lam in aTypes])

    # Define the constraints
    # Attacker
    attackerConstraints = [sum([w[t,d,t,lam] * _aPenalties[lam][t] for t in targetRange for d in defenders]) \
                        + sum([(w2[t,lam] - sum([w[t,d,t,lam] for d in defenders])) * _aRewards[lam][t] for t in targetRange]) \
                        >= sum([w[t,d,tPrime,lam] * _aPenalties[lam][tPrime] for t in targetRange for d in defenders]) \
                        + sum([(w2[t,lam] - sum([w[t,d,tPrime,lam] for d in defenders])) * _aRewards[lam][tPrime] for t in targetRange]) \
                        for lam in aTypes for tPrime in targetRange]
    # Defender
    defenderConstraints = [sum([q[lam] * sum([w[t,d,t,lam] * _dRewards[d][t] for t in targetRange]) for lam in aTypes]) \
                        + sum([q[lam] * sum([(w2[t,lam] - sum([w[t,d,t,lam] for d in defenders])) * _dPenalties[d][t] for t in targetRange]) for lam in aTypes]) \
                        + sum([q[lam] * sum([sum([w[t,d,tPrimePrime,lam] for t in targetRange]) * _dCosts[d][tPrimePrime] for tPrimePrime in targetRange]) for lam in aTypes]) \
                        >= sum([q[lam] * w2[tPrime,lam] * _dRewards[d][tPrime] for lam in aTypes]) \
                        + sum([q[lam] * sum([sum([w[t,dPrime,t,lam] for dPrime in defenders if dPrime != d]) * _dRewards[d][t] for t in targetRange if t != tPrime]) for lam in aTypes]) \
                        + sum([q[lam] * sum([w[t,d,t,lam] * _dPenalties[d][t] for t in targetRange if t != tPrime]) for lam in aTypes]) \
                        + sum([q[lam] * sum([(w2[t,lam] - sum([w[t,dPrime,t,lam] for dPrime in defenders])) * _dPenalties[d][t] for t in targetRange if t != tPrime]) for lam in aTypes]) \
                        + _dCosts[d][tPrime] \
                        for d in defenders for tPrime in targetRange]
    # Proposition constraints
    p11Constraints = [sum([w2[t,lam] for t in targetRange]) == 1 for lam in aTypes]
    p12Constraints = [sum([w[t,d,tPrime,lam] for tPrime in targetRange]) == w2[t,lam] for lam in aTypes for d in defenders for t in targetRange]
    p13Constraints = [sum([w[t,d,tPrime,lam] for d in defenders]) <= w2[t,lam] for lam in aTypes for tPrime in targetRange for t in targetRange]
    # Add the constraints
    attackerConstraints = model.add_constraints(attackerConstraints)
    defenderConstraints = model.add_constraints(defenderConstraints)
    p11Constraints = model.add_constraints(p11Constraints)
    p12Constraints = model.add_constraints(p12Constraints)
    p13Constraints = model.add_constraints(p13Constraints)
    # Solve the model
    model.maximize(objectiveFunction)
    model.solve()
    print(model.get_solve_status())
    return model.solution.get_objective_value(), model, None
