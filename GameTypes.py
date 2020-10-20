from docplex.mp.model import Model
from itertools import permutations
from functools import reduce
from operator import mul
from numpy import argmax
from time import time as getTime
import matplotlib.pyplot as plt
import networkx as nx
import random
import copy

from constants import DEFENDERNUM,ATTACKERNUM,TARGETNUM,AVGCOUNT,M
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
    return model.solution.get_objective_value(), model

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
    objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a,defenders, _dCosts, _dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])

    model.add_constraints([sum([w[(s,a,lam)] * aUtility(s,a,lam, aPenalties, aRewards) for s in placements]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam, aPenalties, aRewards) for s in placements]) for a in attackerActions for b in attackerActions if a != b for lam in aTypes], names=[f"c att {lam} suggested {a}, but goes to {b}" for a in attackerActions for b in attackerActions if a != b for lam in aTypes])
    model.add_constraints([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) >= \
                           sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) \
                           for m in defenders for d in range(targetNum + 1) for e in range(targetNum + 1) if d != e for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model

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
    objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a,defenders, _dCosts, _dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] * aUtility(s,a,lam, aPenalties, aRewards) for s in placements]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam, aPenalties, aRewards) for s in placements]) for a in attackerActions for b in attackerActions if a != b for lam in aTypes], names=[f"c att {lam} suggested {a}, but goes to {b}" for a in attackerActions for b in attackerActions if a != b for lam in aTypes])
    model.add_constraints([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) >= \
                           sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) \
                           for m in defenders for d in range(targetNum + 1) for e in range(targetNum + 1) if d != e for lam in aTypes], names=[f"defender {m} suggested {d}, but goes to {e} with att {lam}" for m in defenders for d in range(targetNum + 1) for e in range(targetNum + 1) if d != e for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes], names=[f"sum must be 1 for att: {lam}" for lam in aTypes])
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model

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
    objectiveFunction = sum([q[lam] * sum([w[(s,a,lam)] * defenderSocialUtility(s,a,defenders, _dCosts, _dPenalties) for s in placements for a in attackerActions]) for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] * aUtility(s,a,lam, aPenalties, aRewards) for s in placements]) >= sum([w[(s,a,lam)] * aUtility(s,b,lam, aPenalties, aRewards) for s in placements]) for a in attackerActions for b in attackerActions if a != b for lam in aTypes], names=[f"c att {lam} suggested {a}, but goes to {b}" for a in attackerActions for b in attackerActions if a != b for lam in aTypes])
    model.add_constraints([sum([q[lam] * sum([w[dm,a,lam] * utilityM(d,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) >= \
                           sum([q[lam] * sum([w[dm,a,lam] * utilityM(e,dm,a,m, _dRewards, _dPenalties, _dCosts) for a in attackerActions for dm in placements if dm[m] == d])]) \
                           for m in defenders for d in range(targetNum + len(defenders)) for e in range(targetNum + len(defenders)) if d != e for lam in aTypes])
    model.add_constraints([sum([w[(s,a,lam)] for s in placements for a in attackerActions]) == 1 for lam in aTypes])
    model.maximize(objectiveFunction)
    model.solve()
    return model.solution.get_objective_value(), model

def solveBPNONDDualEllipsoid(targetNum, defenders, dRewards, dPenalties, dCosts, aTypes, aRewards, aPenalties, q):
    """A game where defender assignments are not allowed to overlap, with as many
    dummy targets as defenders (represents defenders not having to be assigned).
    This problem is the dual of the primal above, and is solved using the ellipsoid
    method."""
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

    # Get the placements that occur with no overlap
    overlapPlacements = getPlacements(defenders, targetNumWithDummies)
    placements = list(filter(lambda x: len(set(x)) == len(x), overlapPlacements))
    # Generate the keys
    aKeys = [(t,tPrime,lam) for t in targetRange for tPrime in targetRange for lam in aTypes]
    bKeys = [(t,tPrime,d) for t in targetRange for tPrime in targetRange for d in defenders]
    # Get a random subset of the placements
    subsetCount = len(defenders)
    subsetS = random.choices(placements, k=subsetCount)

    # Generate the dual model using the limited set of placements
    while True:
        relaxedModel = Model('relaxedModel')
        g = relaxedModel.continuous_var_dict(keys=aTypes, lb=0, ub=1, name="g")
        a = relaxedModel.continuous_var_dict(keys=aKeys, lb=0, ub=1, name="a")
        b = relaxedModel.continuous_var_dict(keys=bKeys, lb=0, ub=1, name="b")
        objectiveFunction = sum([g[lam] for lam in aTypes])

        dualConstraints = relaxedModel.add_constraints([                            \
            sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,sa,lam,_aPenalties,_aRewards)) * a[sa,tPrime,lam] for tPrime in targetRange]) + \
            sum([(utilityM(tPrime,sd,sa,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,sa,d,_dRewards,_dPenalties,_dCosts)) * b[sd[d],tPrime,d]  for d in defenders for tPrime in targetRange]) + \
            g[lam] \
            >= q[lam] * defenderSocialUtility(sd,sa,defenders,_dCosts,_dPenalties)  \
            for sd in subsetS for sa in targetRange for lam in aTypes])
        relaxedModel.minimize(objectiveFunction)
        relaxedModel.solve() # Alpha and Beta have values for each instance of target and attacker
        relaxedModel.export(f"relaxedModel.lp")

        # Determine the violated constraints from the solution s* above (represented
        # by alpha and beta)
        violatedPairs = []
        for t0 in targetRange:
            for lam in aTypes:
                values = [\
                    sum([(aUtility(sd,tPrime,lam,_aPenalties,_aRewards) - aUtility(sd,t0,lam,_aPenalties,_aRewards)) * float(a[t0,tPrime,lam]) for tPrime in targetRange]) + \
                    sum([(utilityM(tPrime,sd,t0,d,_dRewards,_dPenalties,_dCosts) - utilityM(sd[d],sd,t0,d,_dRewards,_dPenalties,_dCosts)) * float(b[sd[d],tPrime,d]) for d in defenders for tPrime in targetRange]) - \
                    (q[lam] * defenderSocialUtility(sd,t0,defenders,_dCosts,_dPenalties))
                    for sd in placements]
                value = min(values)
                # If any are negative, that solution s* violates the constraint represented by lam,t0
                # Add it to the set of problems to solve (9) for.
                if value < 0:
                    violatedPairs.append((lam,t0))

        # If there are no violated constraints, return the solution
        if len(violatedPairs) == 0:
            print("NO VIOLATED CONSTRAINTS -- RETURNING SOLUTION")
            return relaxedModel.solution.get_objective_value(), relaxedModel
        else:
            print("VIOLATED CONSTRAINTS")

        # For each violated constraint lam,t0, split (9) into two subproblems and solve each
        for lam, t0 in violatedPairs:
            # Subproblem 1
            # These are the placements that do not cover t0.
            # For each of these placements s, define a set of weights to make a
            # graph, adding extra defenders as necessary for a maximum bipartite
            # matching problem
            solutions = []
            for sd in placements:
                edges = {}
                # Build the weights
                for d in defenders:
                    edges[f"d_{d}"] = {}
                    for t in targetRange:
                        if t != t0:
                            print(f"t: {t}, d: {d}, t0: {t0}, sd: {sd}, sd[d]: {sd[d]}")
                            weightValue = -q[lam]*_dCosts[d][t] \
                                            + (_dRewards[d][t0] + _dCosts[d][t0] - _dPenalties[d][t0] - _dCosts[d][sd[d]]) * float(b[t, t0, d]) \
                                            + sum([(_dCosts[d][tPrime] - _dCosts[d][t]) * float(b[t,tPrime,d]) for tPrime in targetRange if tPrime != t0]) \
                                            + (_aPenalties[lam][t] - _aRewards[lam][t0]) * float(a[t0, t, lam])
                            edges[f"d_{d}"][f"t_{t}"] = {"weight": weightValue}
                        else:
                            weightValue = M
                            edges[f"d_{d}"][f"t_{t}"] = {"weight": weightValue}
                for d in range(len(defenders), targetNumWithDummies):
                    edges[f"ed_{d}"] = {}
                    for t in targetRange:
                        weightValue = (_aRewards[lam][t] - _aRewards[lam][t0]) * float(a[t0,t,lam])
                        edges[f"ed_{d}"][f"t_{t}"] = {"weight": weightValue}
                print()

                # Build the graph
                G = nx.from_dict_of_dicts(edges)
                # Solve the problem
                matchings = nx.algorithms.bipartite.minimum_weight_full_matching(G)
                # Convert the solution back into our setting and calculate the cost
                cost = 0
                s = [0] * len(defenders)
                for k,v in matchings.items():
                    defender = int(k.split("_")[1])
                    target = int(v.split("_")[1])
                    if k.startswith("d_"):
                        s[defender] = target
                        cost += edges[k][v]["weight"]
                    elif k.startswith("ed_"):
                        cost += edges[k][v]["weight"]
                solutions.append((s,cost))
                # Solve the problem
            print(solutions)
            asdfasdf


            # Subproblem 2
            # Fix each possible defender that coveres t0. For each of these


            # After obtaining solutions to both parts of problem 9, we have two
            # defender placements paired with t0 to give us a new placement (sd)
            # to add to the subset considered in the original. We obtain two such
            # solutions for every violated (lam,t0) pair. Add them all and do the
            # process again.



            # weights = {}
            # for d in defenders:
            #     for t in targetRange:
            #         if t != t0:
            #             pass
            #             # weights[(d,t)] = -q[lam]*_dCosts[d][t] \
            #             #                 + (_dRewards[d][t0] + _dCosts[d][t0] - _dCosts[?][t]) * b[t, t0, d] \
            #             #                 + sum([(_dCosts[d][tPrime] - _dCosts[d][t]) * b[t,tPrime,d] for tPrime in targetRange if tPrime != t0]) \
            #             #                 + (aPenalties[lam][t] - aRewards[lam][t0]) * a[t0, t, lam]
            #     weights[(d,t0)] = M # A huge number: "infinity"
            # # Add in the extra defenders
            # print("THE EXTRAS")
            # for d in range(len(defenders), targetNumWithDummies - 1):
            #     for t in range(targetNumWithDummies):
            #         weights[(d,t)] = (_aRewards[lam][t] - _aRewards[lam][t0]) * float(a[t0,t,lam])
            # print(weights)



            # for d0 in defenders:
            #     for d in defenders:
            #         if d != d0:
            #             for t in targetRange:
            #                 if t != t0:
            #     # Add in the extra defenders
            #     print("THE EXTRAS")
            #     for d in range(len(defenders), targetNumWithDummies - 1):
            #         for t in range(targetNumWithDummies):
            #             weights[(d,t)] = (_aRewards[lam][t] - _aRewards[lam][t0]) * float(a[t0,t,lam])
            #     print(weights2)



    return model.solution.get_objective_value(), model

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
    return baselineUtility, models
