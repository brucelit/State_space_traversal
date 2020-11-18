import heapq
import os

import numpy as np
from pm4py.objects.petri import align_utils as utils
from pm4py.visualization.petrinet import visualizer

from astar_implementation import utilities,heuristic
from astar_implementation import visualization

ret_tuple_as_trans_desc = False


def astar(sync_net, sync_im, sync_fm, cost_function):
    # ---------------Line 1-9----------------

    incidence_matrix, place_index, trans_index = utilities.construct_incident_matrix(sync_net)
    ini_vec, fin_vec, cost_vec = utilities.vectorize_initial_final_cost(place_index, trans_index, sync_im, sync_fm,
                                                                        cost_function)
    ini_tuple = tuple(ini_vec)
    closed_set = set()
    heuristic_set = set()
    f_score = {}
    h_score = {}
    g_score = {}
    enabled_trans = []
    for trans in trans_index:
        enabled_trans.append(trans)
    h_score[ini_tuple], solution_x = utilities.compute_estimated_heuristic(ini_vec, fin_vec, incidence_matrix, cost_vec)
    g_score[ini_tuple] = 0
    f_score[ini_tuple] = h_score[ini_tuple]
    ini_state = utilities.State(f_score[ini_tuple], 0, h_score[ini_tuple], None, sync_im, ini_tuple, None, solution_x)
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0
    consumption_matrix = utilities.construct_consumption_matrix(sync_net)
    viz, places_sort_list = visualization.graphviz_visualization(sync_net, image_format="png", initial_marking=sync_im,
                                                                 final_marking=sync_fm, current_marking=None)
    # h1 = ch.test_estimated_heuristic(incidence_matrix,consumption_matrix,place_index,trans_index,sync_net,cost_vec,sync_im,sync_fm)
    # ----------Line 10------------
    while len(open_set) > 0:
        curr = heapq.heappop(open_set)
        current_marking = curr.m
        # new_viz = visualization.graphviz_state_change(viz, places_sort_list, current_marking)
        # new_viz.graph_attr['label'] = "F-score for current state: "+str(curr.f)+ "\nNumber of states visited: "+str(visited)+"\nNumber of states in open set: "+str(len(open_set))
        # visualizer.save(new_viz, os.path.join("E:/Thesis/img", "step"+str(visited) + ".png"))
        curr_vec = utilities.encode_marking(current_marking, place_index)
        if curr_vec == fin_vec:
            result = utilities.reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc)
            # new_viz.graph_attr['label'] = "Optimal alignment: "+ str(result["alignment"])+ "\nCost of optimal alignment: "+ str(result["cost"])+ "\nNumber of states visited: "+ str(result["visited_states"])
            # visualizer.save(new_viz, os.path.join("E:/Thesis/img", "step"+str(visited) + ".png"))
            print("Optimal alignment:", result["alignment"], "\nCost of optimal alignment:",
                  result["cost"], "\nNumber of states visited:", result["visited_states"],
                  "\nqueued states:",result["queued"],
                  "\ntraversed arcs:",result["traversed"])
            return result
        if curr in heuristic_set:
            heuristic_set.remove(curr)
            h_est, x = utilities.compute_estimated_heuristic(curr_vec, fin_vec, incidence_matrix, cost_vec)

            if h_est > curr.h:
                curr.f = curr.g + h_est
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
            if curr.t is None:
                break
            if curr.t.label[0] == ">>" and t.label[1] == ">>":
                violated_trans.append(t)
        for t in violated_trans:
            enabled_trans.remove(t)
        enabled_trans = sorted(enabled_trans, key=lambda k: k.label,reverse=True)
        for t in enabled_trans:
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)
            new_vec = utilities.encode_marking(new_marking, place_index)
            new_tuple = tuple(new_vec)
            t_index = trans_index[t]
            new_h_score, new_solution_x, h_trust = utilities.compute_exact_heuristic(curr.h, curr.solution_x, cost_vec, t_index)
            print(t, new_h_score, curr.h, h_trust)
            if new_tuple not in g_score:
                g_score[new_tuple] = 100000
            if new_marking not in closed_set:
                new_state = utilities.State(0, 0, 0, t, new_marking, new_tuple, curr, new_solution_x)
                if curr.g + cost_function[t] < g_score[new_tuple]:
                    new_state.g = curr.g + cost_function[t]
                    queued += 1
                    new_state.h = curr.h - cost_function[t]
                    new_state.f = new_state.g + new_h_score
                    g_score[new_tuple] = new_state.g
                    h_score[new_tuple] = new_state.h
                    f_score[new_tuple] = new_state.f
                    if not h_trust:
                        heuristic_set.add(new_state)
                    heapq.heappush(open_set, new_state)
