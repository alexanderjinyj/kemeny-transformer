import numpy as np
from itertools import combinations, permutations
from time import time

import gurobipy as gp
from gurobipy import GRB, quicksum

from kemeny_transformer.utils.graph import build_graph_margin as build_graph


def aggregate_kemeny(n_voters, n_candidates, ranks):
    """
    Use Gurobi solver to find the Kemeny-optimal ranking by solving
    a binary integer programming problem with pairwise and transitivity constraints.

    Args:
        n_voters: int, number of voters
        n_candidates: int, number of candidates
        ranks: numpy array of shape (n_voters, n_candidates)
    Returns:
        aggr_rank: numpy array of aggregated ranking scores, or None if infeasible
        runtime: float, solver runtime in seconds
    """
    # Declare gurobi model object
    m = gp.Model("aggregate_kemeny")
    m.setParam("OutputFlag", 0)
    m.setParam("Threads", 16)

    # Indicator variable for each pair
    x = {}
    c = 0
    for i in range(n_candidates):
        for j in range(n_candidates):
            x[c] = m.addVar(vtype=GRB.BINARY, name="x(%d)(%d)" % (i, j))
            c += 1
    m.update()

    idx = lambda i, j: n_candidates * i + j

    # pairwise constraints
    for i, j in combinations(range(n_candidates), 2):
        m.addConstr(x[idx(i, j)] + x[idx(j, i)] == 1)
    m.update()

    # transitivity constraints
    for i, j, k in permutations(range(n_candidates), 3):
        m.addConstr(x[idx(i, j)] + x[idx(j, k)] + x[idx(k, i)] >= 1)
    m.update()

    # Set objective
    # maximize c.T * x
    edge_weights = build_graph(ranks)
    c = -1 * edge_weights.ravel()
    m.setObjective(quicksum(c[i]*x[i] for i in range(len(x))), GRB.MAXIMIZE)
    m.update()

    t0 = time()
    m.optimize()
    t1 = time()

    if m.status == GRB.OPTIMAL:
        # get consensus ranking
        sol = []
        for i in x:
            sol.append(x[i].X)
        sol = np.array(sol)
        aggr_rank = np.sum(sol.reshape((n_candidates, n_candidates)), axis=1)
        return aggr_rank, t1 - t0
    else:
        return None, t1 - t0
