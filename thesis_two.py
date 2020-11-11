import heapq

import numpy as np
from pm4py.objects import petri
from pm4py.objects.petri import align_utils as utils
from pm4py.objects.petri.utils import decorate_places_preset_trans, decorate_transitions_prepostset

from astar_implementation import utilities as utilities

ret_tuple_as_trans_desc = False


def astar_with_split(sync_net, sync_im, sync_fm, cost_function, split_list):
    # ---------------Line 1-9----------------
    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)
    incidence_matrix = petri.incidence_matrix.construct(sync_net)
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(incidence_matrix, sync_im, sync_fm, cost_function)
    ini_tuple = tuple(ini_vec)
    closed_set = set()
    heuristic_set = set()
    s = 0
    cost_vec = [x * 1.0 for x in cost_vec]
    f_score = {}
    h_score = {}
    g_score = {}
    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    h_score[ini_tuple], solution_x = utilities.compute_estimated_heuristic(ini_vec, fin_vec, a_matrix, cost_vec, split_list)
    g_score[ini_tuple] = 0
    f_score[ini_tuple] = h_score[ini_tuple]
    ini_state = utilities.State(f_score[ini_tuple], 0, h_score[ini_tuple], None, sync_im, ini_tuple, None, solution_x)
    open_set = [ini_state]
    heapq.heapify(open_set)
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)
    visited = 0
    queued = 0
    traversed = 0
    # ----------Line 10------------
    while len(open_set) > 0:
        curr = heapq.heappop(open_set)
        current_marking = curr.m
        curr_vec = incidence_matrix.encode_marking(current_marking)
        if curr_vec == fin_vec:
            result = utilities.reconstruct_alignment(curr, visited, queued, traversed,
                                                         ret_tuple_as_trans_desc=ret_tuple_as_trans_desc)
            print("Optimal alignment:", result["alignment"], "\nCost of optimal alignment：",
                  result["cost"], "\nNumber of states visited:", result["visited_states"])
            return result

        #s which represents the index of the last event explained by the closed markings in set A
        if curr in heuristic_set:
            heuristic_set.remove(curr)
            if s not in split_list:
                split_list.add(s)
                return astar_with_split(sync_net, sync_im, sync_fm, cost_function, split_list)
            h_est, x = utilities.compute_estimated_heuristic(curr_vec, fin_vec, a_matrix, cost_vec)
            if h_est > curr.h:
                curr.f = curr.g + h_est
                continue

        closed_set.add(curr.m)
        visited += 1
        s = max(s,incidence_matrix.transitions[curr.t])


        # ----------Line 31-48 ----------
        possible_enabling_transitions = set()
        for p in current_marking:
            for t in p.ass_trans:
                possible_enabling_transitions.add(t)
        enabled_trans = [t for t in possible_enabling_transitions if t.sub_marking <= current_marking]
        # for t in enabled_trans:
        #     if curr.t == None:
        #         break
        #     if curr.t.label[1] == ">>" and t.label[0] == ">>":
        #         enabled_trans.remove(t)

        for t in enabled_trans:
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)
            new_vec = incidence_matrix.encode_marking(new_marking)
            new_tuple = tuple(new_vec)
            t_index = incidence_matrix.transitions[t]
            new_h_score, h_trust, new_solution_x = utilities.compute_exact_heuristic(solution_x, cost_vec, t_index)
            if new_tuple not in g_score:
                g_score[new_tuple] = 100000
            if new_marking not in closed_set:
                new_state = utilities.State(0, 0, 0, t, new_marking, new_tuple, curr, new_solution_x)
                if curr.g + cost_function[t] < g_score[new_tuple]:
                    new_state.g = curr.g + cost_function[t]
                    queued += 1
                    # new_state.h = curr.h - cost_function[t]
                    if h_trust:
                        new_state.f = new_state.g + new_h_score
                    else:
                        heuristic_set.add(new_state)
                        new_state.f = new_state.g + curr.h - cost_function[t]
                    g_score[new_tuple] = new_state.g
                    h_score[new_tuple] = new_state.h
                    f_score[new_tuple] = new_state.f
                    heapq.heappush(open_set, new_state)