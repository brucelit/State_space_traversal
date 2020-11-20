import heapq
import os
import copy

import numpy as np
from pm4py.objects import petri
from pm4py.objects.petri import align_utils as utils

from astar_implementation import utilities as utilities
from astar_implementation import heuristic,util
from astar_implementation import visualization
from pm4py.visualization.petrinet import visualizer
ret_tuple_as_trans_desc = False


def astar_with_split(sync_net, sync_im, sync_fm, consumption_matrix, incidence_matrix, p_index, t_index, cost_function, split_lst, visited):
    # ---------------Line 1-9----------------
    ini_vec, fin_vec, cost_vec = utilities.vectorize_initial_final_cost(p_index, t_index, sync_im, sync_fm, cost_function)
    ini_tuple = tuple(ini_vec)
    closed_set = set()
    heuristic_set = set()
    cost_vec = [x * 1.0 for x in cost_vec]
    g_score_set = {}
    enabled_trans = []
    for trans in t_index:
        enabled_trans.append(trans)
    ini_h, parikh_vector = heuristic.compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, split_lst, t_index)
    print("ini_h",ini_h)
    g_score_set[ini_tuple] = 0
    ini_f = ini_h
    pre_trans_lst = []
    ini_state = util.State(ini_f, 0, ini_h, sync_im, ini_tuple, None, None, pre_trans_lst, None, parikh_vector,)
    open_set = [ini_state]
    heapq.heapify(open_set)
    queued = 0
    traversed = 0
    # viz, places_sort_list = visualization.graphviz_visualization(sync_net, image_format="png",
    #                         initial_marking=sync_im, final_marking=sync_fm, current_marking=None)

    # ----------Line 10------------
    while len(open_set) > 0:
        curr = heapq.heappop(open_set)
        current_marking = curr.marking
        # print(current_marking, "previous trans", curr.pre_transition)
        # new_viz = visualization.graphviz_state_change(viz, places_sort_list, current_marking)
        # new_viz.graph_attr['label'] = "Number of split: " + str(len(split_lst)-1) +"\nTransition in split set: " + \
        #                               str(split_lst[1:]) +"\nF-score for current state: " + \
        #                               str(curr.f) + "\nNumber of states visited: " + \
        #                               str(visited) + "\nNumber of states in open set: " + \
        #                               str(len(open_set))
        # visualizer.save(new_viz, os.path.join("E:/Thesis/img", "step" + str(visited) + ".png"))
        curr_vec = utilities.encode_marking(current_marking, p_index)
        if curr_vec == fin_vec:
            result = utilities.reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc)
            # new_viz.graph_attr['label'] = "Number of split: " + str(len(split_lst)-1) + "\nTransition in split set: " + \
            #                               str(split_lst[1:]) + "\nOptimal alignment: " + \
            #                               str(result["alignment"]) + "\nCost of optimal alignment: " + \
            #                               str(result["cost"]) + "\nNumber of states visited: " + \
            #                               str(result["visited_states"])
            # visualizer.save(new_viz, os.path.join("E:/Thesis/img", "step" + str(visited) + ".png"))
            print(curr.pre_trans_lst)
            print("Optimal alignment:", result["alignment"], "\nCost of optimal alignment:",
                  result["cost"], "\nNumber of states visited:", result["visited_states"],"\nNumber of split: " + str(len(split_lst)-1) +"\nTransition in split set: " + \
                                      str(split_lst[1:]) +"\nF-score for final state: " + \
                                      str(curr.f) + "\nNumber of states visited: "+str(visited))
            return result
        if curr in heuristic_set:
            if curr.last_sync not in split_lst:
                print(current_marking,curr.pre_trans_lst,"new last sync:", curr.last_sync)
                split_lst.append(curr.last_sync)
                return astar_with_split(sync_net, sync_im, sync_fm, consumption_matrix, incidence_matrix,
                                        p_index, t_index, cost_function, split_lst, visited)
            new_heuristic, curr.parikh_vector = heuristic.compute_exact_heuristic(curr_vec, fin_vec, incidence_matrix, cost_vec)
            # print("transition before",curr.t,"exact h for marking:", new_heuristic, "old h:",curr.h)
            heuristic_set.remove(curr)
            if new_heuristic > curr.h:
                curr.f = curr.g + new_heuristic
                curr.h = new_heuristic
                continue
        closed_set.add(curr)
        visited += 1
        possible_enabling_transitions = set()
        for p in current_marking:
            for t in p.ass_trans:
                possible_enabling_transitions.add(t)
        enabled_trans = [t for t in possible_enabling_transitions if t.sub_marking <= current_marking]
        violated_trans = []
        for t in enabled_trans:
            if curr.pre_transition is None:
                break
            if curr.pre_transition.label[0] == ">>" and t.label[1] == ">>":
                violated_trans.append(t)
        for t in violated_trans:
            enabled_trans.remove(t)
        enabled_trans = sorted(enabled_trans, key=lambda k: k.label)
        for t in enabled_trans:
            traversed += 1
            new_pre_trans_lst = copy.deepcopy(curr.pre_trans_lst)
            new_pre_trans_lst.append(t)
            new_marking = utils.add_markings(current_marking, t.add_marking)
            new_vec = utilities.encode_marking(new_marking, p_index)
            new_tuple = tuple(new_vec)
            if new_tuple not in g_score_set:
                g_score_set[new_tuple] = 100000
            if new_marking not in closed_set:
                new_h_score, new_parikh_vector, h_trust = heuristic.compute_estimate_heuristic(curr.h, curr.parikh_vector,
                                                                                               t_index[t],cost_vec)
                if t != None and (t.label[0] == t.label[1]):
                    new_state = util.State(0, 0, 0, new_marking, new_tuple, curr, t, new_pre_trans_lst, t, new_parikh_vector)
                else:
                    new_state = util.State(0, 0, 0, new_marking, new_tuple, curr, t, new_pre_trans_lst,curr.last_sync, new_parikh_vector)
                # compute cost so far
                a = curr.g + cost_function[t]
                if a < g_score_set[new_tuple]:
                    new_state.g = a
                    queued += 1
                    if not h_trust:
                        new_state.h = new_h_score
                        heuristic_set.add(new_state)
                        new_state.f = a + curr.h - cost_vec[t_index[t]]
                    else:
                        new_state.h = curr.h - cost_vec[t_index[t]]
                        new_state.f = new_state.g + new_h_score
                    g_score_set[new_tuple] = new_state.g
                    heapq.heappush(open_set, new_state)