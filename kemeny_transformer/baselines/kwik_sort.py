from math import floor, ceil

import numpy as np

from random import randrange

from kemeny_transformer.utils.graph import build_graph_counts as build_graph


def KwikSort(vertices, rankings, initial_pivot=None):
    Fas_Tournament = build_graph(rankings)
    unweighted_majority_Tournament = generate_unweighted_majority_Tournament(vertices, Fas_Tournament)
    if initial_pivot is None:
        initial_pivot = np.random.choice(vertices)
    ranking_opt = Kwik_Sort_recur(vertices, unweighted_majority_Tournament, initial_pivot, Fas_Tournament)

    return ranking_opt

def Kwik_Sort_recur(vertices, unweighted_majority_Tournament, pivot, Fas_Tournament):
    Arc = unweighted_majority_Tournament

    if len(vertices) == 0:
        return np.zeros(0)
    size_candidates = len(vertices)
    vertices_L = []
    vertices_R = []
    for j in range(size_candidates):
        if Arc[vertices[j], pivot] == 1:
            vertices_L.append(vertices[j])
        if Arc[pivot, vertices[j]] == 1:
            vertices_R.append(vertices[j])
    vertices_L = np.array(vertices_L)
    vertices_R = np.array(vertices_R)
    if len(vertices_L) > 0:
        pivot_L = np.random.choice(vertices_L)
    else:
        pivot_L = 0
    if(len(vertices_R) > 0):
        pivot_R = np.random.choice(vertices_R)
    else:
        pivot_R = 0
    permutation_L = Kwik_Sort_recur(vertices_L, unweighted_majority_Tournament, pivot_L, Fas_Tournament).astype(int)
    permutation_R = Kwik_Sort_recur(vertices_R, unweighted_majority_Tournament, pivot_R, Fas_Tournament).astype(int)
    return np.concatenate((permutation_L, np.array([pivot]).astype(int), permutation_R))


def generate_unweighted_majority_Tournament(Verticles, Fas_Tournament):
    size_candidates = Fas_Tournament.shape[0]
    size_vertices = len(Verticles)
    # This creates arcs where there is a clear winner.
    strict_wins = (Fas_Tournament > Fas_Tournament.T).astype(int)
    # This creates a symmetric matrix where ties are 1, others are 0.
    ties = (Fas_Tournament == Fas_Tournament.T).astype(int)
    # Use the "upper triangle" to break ties
    tie_breaker = np.triu(ties, k=1)
    Arc = strict_wins + tie_breaker
    return Arc
