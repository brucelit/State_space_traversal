import heapq
import os
import copy

from pm4py.objects.petri import align_utils as utils

from astar_implementation import utilities as utilities
from astar_implementation import heuristic,util
from astar_implementation import visualization
from pm4py.visualization.transition_system import visualizer as ts_visualizer
ret_tuple_as_trans_desc = False


def astar_with_split(max_num, order, ts, place_map, sync_net, sync_im, sync_fm, consumption_matrix, incidence_matrix, p_index, t_index, cost_function, split_lst, visited):

    # ---------------Line 1-9----------------
    ini_vec, fin_vec, cost_vec = utilities.vectorize_initial_final_cost(p_index, t_index, sync_im, sync_fm, cost_function)
    ini_tuple = tuple(ini_vec)
    closed_set = set()
    heuristic_set = set()
    cost_vec = [x * 1.0 for x in cost_vec]
    g_score_set = {}
    ini_h, parikh_vector = heuristic.compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, split_lst, t_index)
    g_score_set[ini_tuple] = 0
    ini_f = ini_h
    pre_trans_lst = []
    ini_state = util.State(0, ini_f, 0, ini_h, sync_im, ini_tuple, None, None, pre_trans_lst, None, parikh_vector)
    open_set = [ini_state]
    heapq.heapify(open_set)
    queued = 0
    traversed = 0
    valid_state_lst = []
    invalid_state_lst = []

    new_split_point = None
    # ----------Line 10------------
    while len(open_set) > 0:
        visited += 1
        order += 1
        curr = heapq.heappop(open_set)
        current_marking = curr.marking
        print("\nstep",visited,"\ncurrent marking:", curr.marking, curr.pre_trans_lst)

        element_curr_state = []
        for p in sorted(curr.marking, key=lambda k: k.name[1]):
            element_curr_state.append(place_map[p])
        lst2 = '[%s]' % ', '.join(map(str, element_curr_state))
        lst3 = []
        lst3.append(lst2)
        curr_state = '[%s]' % ', '.join(map(str, lst3))
        gviz = visualization.viz_state_change(ts, curr_state, valid_state_lst, invalid_state_lst)
        gviz.graph_attr['label'] = "\nNumber of states visited: "+str(visited) + "\nNumber of states in open set: "+str(len(open_set)) + "\nsplit list:" + str(split_lst[1:])
        ts_visualizer.save(gviz, os.path.join("E:/Thesis/img", "step" + str(order)+ ".png"))
        curr_vec = utilities.encode_marking(current_marking, p_index)
        element_valid_state = []
        for p in sorted(curr.marking, key=lambda k: k.name[1]):
            element_valid_state.append(place_map[p])
        lst2 = '[%s]' % ', '.join(map(str, element_valid_state))
        if lst2 not in valid_state_lst:
            valid_state_lst.append(lst2)
        if curr_vec == fin_vec:
            result = utilities.reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc)
            print("Optimal alignment:", result["alignment"], "\nCost of optimal alignment:",
                  result["cost"], "\nNumber of states visited:", result["visited_states"],"\nNumber of split: " + str(len(split_lst)-1) +"\nTransition in split set: " + \
                                      str(split_lst[1:]) +"\nF-score for final state: " + \
                                      str(curr.f) + "\nNumber of states visited: "+str(visited))
            return result

        if curr in heuristic_set:
            # split_point = curr.last_sync
            if new_split_point not in split_lst:
                split_lst.append(new_split_point)
                print("new split list", split_lst)
                return astar_with_split(max_num, order, ts, place_map, sync_net, sync_im, sync_fm, consumption_matrix, incidence_matrix,
                                        p_index, t_index, cost_function, split_lst, visited)
            new_heuristic, curr.parikh_vector = heuristic.compute_exact_heuristic(curr_vec, fin_vec, incidence_matrix, cost_vec)
            heuristic_set.remove(curr)

            if new_heuristic > curr.h:
                curr.f = curr.g + new_heuristic
                curr.h = new_heuristic
                # requeue the state after recalculating
                curr.trust = 0
                heapq.heappush(open_set, curr)
                continue
        closed_set.add(curr)
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

        temp_split_point = None
        j = 0
        # get the split point
        for i in range(len(curr.pre_trans_lst)):
            if curr.pre_trans_lst[i] != None:
                if curr.pre_trans_lst[i].label[0] == curr.pre_trans_lst[i].label[1]:
                    j = j+1
                    temp_split_point = curr.pre_trans_lst[i]
        print("maximum number of trace explained", j)
        if j > max_num:
            new_split_point = temp_split_point
            print("new split point", new_split_point)
            print("index of new split",t_index[new_split_point])
            max_num = j
            print("new j:",j)

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
                    new_state = util.State(h_trust, 0, 0, 0, new_marking, new_tuple, curr, t, new_pre_trans_lst, t, new_parikh_vector)
                else:
                    new_state = util.State(h_trust, 0, 0, 0, new_marking, new_tuple, curr, t, new_pre_trans_lst,curr.last_sync, new_parikh_vector)
                # compute cost so far
                a = curr.g + cost_function[t]
                if a < g_score_set[new_tuple]:
                    new_state.g = a
                    queued += 1
                    element_valid_state = []
                    for p in sorted(new_state.marking, key=lambda k: k.name[1]):
                        element_valid_state.append(place_map[p])
                    lst2 = '[%s]' % ', '.join(map(str, element_valid_state))
                    if h_trust == 1:
                        new_state.h = new_h_score
                        heuristic_set.add(new_state)
                        new_state.f = a + curr.h - cost_vec[t_index[t]]
                        invalid_state_lst.append(lst2)
                    else:
                        new_state.h = curr.h - cost_vec[t_index[t]]
                        new_state.f = new_state.g + new_h_score
                        valid_state_lst.append(lst2)
                    g_score_set[new_tuple] = new_state.g
                    heapq.heappush(open_set, new_state)
                    # print("new state f:", new_state.marking, new_state.f, new_state.g, new_state.h,new_state.trust)
        # print("valid state:",valid_state_lst)
        # print("invalid state:",invalid_state_lst)
        # print("split point", split_point)
        # print("split list", split_lst)
        element_curr_state = []
        for p in sorted(curr.marking, key = lambda k: k.name[1]):
            element_curr_state.append(place_map[p])
        lst2 = '[%s]' % ', '.join(map(str, element_curr_state))
        lst3 = []
        lst3.append(lst2)
        curr_state = '[%s]' % ', '.join(map(str, lst3))
        gviz = visualization.viz_state_change(ts, curr_state, valid_state_lst, invalid_state_lst)
        gviz.graph_attr['label'] = "\nNumber of states visited: "+str(visited) + "\nNumber of states in open set: "+str(len(open_set)) + "\nsplit list:" + str(split_lst[1:])
        ts_visualizer.save(gviz, os.path.join("E:/Thesis/img", "step" + str(order)+ ".png"))

