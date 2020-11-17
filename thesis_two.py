import heapq
import os

import numpy as np
from pm4py.objects import petri
from pm4py.objects.petri import align_utils as utils

from astar_implementation import utilities as utilities
from astar_implementation import heuristic
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
    f_score = {}
    h_score = {}
    g_score = {}
    enabled_trans = []
    for trans in t_index:
        enabled_trans.append(trans)
    h_score[ini_tuple], solution_x = heuristic.ini_exact_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, split_lst, t_index)
    g_score[ini_tuple] = 0
    f_score[ini_tuple] = h_score[ini_tuple]
    ini_state = utilities.State(f_score[ini_tuple], 0, h_score[ini_tuple], None, sync_im, ini_tuple, None,
                                solution_x, last_sync = None)
    open_set = [ini_state]
    heapq.heapify(open_set)
    queued = 0
    traversed = 0
    viz, places_sort_list = visualization.graphviz_visualization(sync_net, image_format="png",
                                                                 initial_marking=sync_im,
                                                                 final_marking=sync_fm, current_marking=None)
    # h1 = ch.test_estimated_heuristic(incidence_matrix,consumption_matrix,place_index,trans_index,sync_net,cost_vec,sync_im,sync_fm)
    # ----------Line 10------------
    while len(open_set) > 0:
        curr = heapq.heappop(open_set)
        current_marking = curr.m
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
            print("Optimal alignment:", result["alignment"], "\nCost of optimal alignment:",
                  result["cost"], "\nNumber of states visited:", result["visited_states"],"\nNumber of split: " + str(len(split_lst)-1) +"\nTransition in split set: " + \
                                      str(split_lst[1:]) +"\nF-score for final state: " + \
                                      str(curr.f) + "\nNumber of states visited: "+str(visited))
            return result
        if curr in heuristic_set:
            if curr.last_sync not in split_lst:
                split_lst.append(curr.last_sync)
                return astar_with_split(sync_net, sync_im, sync_fm, consumption_matrix, incidence_matrix, p_index, t_index, cost_function,split_lst,visited)
            new_heuristic, curr.solution_x = heuristic.curr_exact_heuristic(curr_vec, fin_vec, incidence_matrix, cost_vec)
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
        for t in enabled_trans:
            if curr.t is None:
                break
            if curr.t.label[0] == ">>" and t.label[1] == ">>":
                enabled_trans.remove(t)
        print("enabled_trans",enabled_trans)
        newlist = sorted(enabled_trans, key=lambda k: k.label)
        print("newlist",newlist)
        for t in newlist:
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)
            new_vec = utilities.encode_marking(new_marking, p_index)
            new_tuple = tuple(new_vec)
            new_h_score, new_solution_x, h_trust = heuristic.estimate_heuristic(curr.h, curr.solution_x, t_index[t], cost_vec)
            if new_tuple not in g_score:
                g_score[new_tuple] = 100000
            if new_marking not in closed_set:
                #用t_index[t]来表示最后的那个index的坐标
                if t.label[0] != ">>" and t.label[1] != ">>" and (t.label[0] != "a" and t.label[1] != "a"):
                    # print("new t",t)
                    new_state = utilities.State(0, 0, 0, t, new_marking, new_tuple, curr, new_solution_x, t)
                else:
                    new_state = utilities.State(0, 0, 0, t, new_marking, new_tuple, curr, new_solution_x, curr.last_sync)
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