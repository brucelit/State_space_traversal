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
    """
    ------------
    # Line 1-10
    ------------
     """

    ini_vec, fin_vec, cost_vec = utilities.vectorize_initial_final_cost(p_index, t_index, sync_im, sync_fm, cost_function)
    closed_set = set()          # init close set
    heuristic_set = set()       # init estimated heuristic set

    # initialize initial state
    ini_state = init_state(sync_im, ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, split_lst, t_index)
    open_set = [ini_state]
    heapq.heapify(open_set)

    # use g_score_set to indicate whether a state has been visited.
    g_score_set = {}
    g_score_set[ini_state.marking_tuple] = 0

    # init the number of states explained  ???????
    s = 0
    new_split_point = None

    # matrice for measurement
    queued = 0
    traversed = 0
    valid_state_lst = []
    invalid_state_lst = []

    '''
    ------------------
    # Line 10-30  
    ------------------
    '''
    while len(open_set) > 0:
        visited += 1
        order += 1

        #  favouring markings for which the exact heuristic is known.
        curr = heapq.heappop(open_set)
        current_marking = curr.marking
        # print("\nstep",visited,"\ncurrent marking:", curr.marking, curr.pre_trans_lst)

        # tranform places in current state to list in form p1,p2,p3…
        curr_state_lst = []
        curr_state_lst.append(marking_to_list(curr, place_map))

        # visualize and save change
        gviz = visualization.viz_state_change(ts, curr_state_lst, valid_state_lst, invalid_state_lst)
        gviz.graph_attr['label'] = "\nNumber of states visited: "+str(visited) + "\nsplit list:" + str(split_lst[1:]) +"\nNumber of states in open set: "+str(len(open_set))
        ts_visualizer.save(gviz, os.path.join("E:/Thesis/img", "step" + str(order)+ ".png"))

        curr_vec = utilities.encode_marking(current_marking, p_index)

        lst2 = marking_to_list(curr, place_map)
        if lst2 not in valid_state_lst:
            valid_state_lst.append(lst2)

        if curr_vec == fin_vec:
            result = print_result(curr, visited, queued, traversed, split_lst)
            return result

        if curr in heuristic_set:
            if new_split_point not in split_lst:
                split_lst.append(new_split_point)
                print("new split list", split_lst)

                # restart with a longer list of split points
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
        # add marking to closed set
        closed_set.add(curr.marking_tuple)
        # keep track of the maximum number of events explained
        new_split_point, max_num = max_events_explained(curr, max_num, t_index)

        '''
        -------------
        # Line 30-end  
        -------------
        '''
        enabled_trans = compute_enabled_transition(curr)
        for t in enabled_trans:
            new_marking = utils.add_markings(current_marking, t.add_marking)
            new_tuple = tuple(utilities.encode_marking(new_marking, p_index))

            # reach a marking not yet visited, or found shorter path
            if new_tuple not in g_score_set:
                g_score_set[new_tuple] = 100000

            if new_tuple not in closed_set:
                traversed += 1
                # create new state
                new_state = compute_new_state(curr, closed_set, cost_vec, t, g_score_set, p_index, t_index)
                a = curr.g + cost_vec[t_index[t]]
                if a < g_score_set[new_tuple]:
                    queued += 1
                    new_state.g = a
                    # compute cost so far
                    lst2 = marking_to_list(new_state, place_map)
                    if new_state.trust == 1:
                        heuristic_set.add(new_state)
                        invalid_state_lst.append(lst2)
                    else:
                        valid_state_lst.append(lst2)
                    g_score_set[new_tuple] = new_state.g
                    heapq.heappush(open_set, new_state)
                    # print("new state f:", new_state.marking, new_state.f, new_state.g, new_state.h,new_state.trust)
        # print("valid state:",valid_state_lst)
        # print("invalid state:",invalid_state_lst)
        # print("split point", split_point)
        # print("split list", split_lst)

        gviz = visualization.viz_state_change(ts, curr_state_lst, valid_state_lst, invalid_state_lst)
        gviz.graph_attr['label'] = "\nNumber of states visited: "+str(visited) + "\nsplit list:" + str(split_lst[1:])+"\nNumber of states in open set: "+str(len(open_set))
        ts_visualizer.save(gviz, os.path.join("E:/Thesis/img", "step" + str(order)+ ".png"))


def init_state(sync_im, ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, split_lst, t_index):
    ini_h, ini_parikh_vector = heuristic.compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix,
                                                           consumption_matrix, split_lst, t_index)
    ini_tuple = tuple(ini_vec)
    ini_f = ini_h
    pre_trans_lst = []

    # not_trust of value 0 means the solution vector is known.
    not_trust = 0
    return util.State(not_trust, ini_f, 0, ini_h, sync_im, ini_tuple, None, None, pre_trans_lst, None, ini_parikh_vector)


def compute_new_state(curr, closed_set, cost_vec, t, g_score_set, p_index, t_index):
    new_pre_trans_lst = copy.deepcopy(curr.pre_trans_lst)
    new_pre_trans_lst.append(t)
    new_marking = utils.add_markings(curr.marking, t.add_marking)
    new_vec = utilities.encode_marking(new_marking, p_index)
    new_tuple = tuple(new_vec)
    if new_marking not in closed_set:
        new_h_score, new_parikh_vector, not_trust = heuristic.compute_estimate_heuristic(curr.h, curr.parikh_vector,
                                                                                       t_index[t], cost_vec)
        if t != None and (t.label[0] == t.label[1]):
            new_state = util.State(not_trust, 0, 0, 0, new_marking, new_tuple, curr, t, new_pre_trans_lst, t,
                                   new_parikh_vector)
        else:
            new_state = util.State(not_trust, 0, 0, 0, new_marking, new_tuple, curr, t, new_pre_trans_lst, curr.last_sync,
                                   new_parikh_vector)
        # compute cost so far
        a = curr.g + cost_vec[t_index[t]]
        if a < g_score_set[new_tuple]:
            new_state.g = a
            if not_trust == 1:
                new_state.h = new_h_score
                new_state.f = a + curr.h - cost_vec[t_index[t]]
            else:
                new_state.h = curr.h - cost_vec[t_index[t]]
                new_state.f = new_state.g + new_h_score
    return new_state


def marking_to_list(curr, place_map):
    lst1 = []
    for p in sorted(curr.marking, key=lambda k: k.name[1]):
        lst1.append(place_map[p])
    lst2 = '[%s]' % ', '.join(map(str, lst1))
    print(lst2)
    lst3 = []
    lst3.append(lst2)
    curr_state_lst = '[%s]' % ', '.join(map(str, lst3))
    return curr_state_lst


def marking_to_list(state, place_map):
    element_valid_state = []
    for p in sorted(state.marking, key=lambda k: k.name[1]):
        element_valid_state.append(place_map[p])
    lst2 = '[%s]' % ', '.join(map(str, element_valid_state))
    return lst2


def compute_enabled_transition(state):
    possible_enabling_transitions = set()
    for p in state.marking:
        for t in p.ass_trans:
            possible_enabling_transitions.add(t)
    enabled_trans = [t for t in possible_enabling_transitions if t.sub_marking <= state.marking]
    violated_trans = []
    for t in enabled_trans:
        if state.pre_transition is None:
            break
        if state.pre_transition.label[0] == ">>" and t.label[1] == ">>":
            violated_trans.append(t)
    for t in violated_trans:
        enabled_trans.remove(t)
    return sorted(enabled_trans, key=lambda k: k.label)


def print_result(state, visited, queued, traversed, split_lst):
    result = utilities.reconstruct_alignment(state, visited, queued, traversed, False)
    print("Optimal alignment:", result["alignment"], "\nCost of optimal alignment:",
          result["cost"], "\nNumber of states visited:", result["visited_states"],
          "\nNumber of split: " + str(len(split_lst) - 1) + "\nTransition in split set: " + \
          str(split_lst[1:]) + "\nF-score for final state: " + \
          str(state.f) + "\nNumber of states visited: " + str(visited))
    return result


def max_events_explained(curr_state, max_num, t_index):
    temp_split_point = None
    new_split_point = None
    j = 0
    # get the split point
    for i in range(len(curr_state.pre_trans_lst)):
        if curr_state.pre_trans_lst[i] != None:
            if curr_state.pre_trans_lst[i].label[0] == curr_state.pre_trans_lst[i].label[1]:
                j = j + 1
                temp_split_point = curr_state.pre_trans_lst[i]
    if j > max_num:
        new_split_point = temp_split_point
        print("new split point", new_split_point)
        max_num = j
    return new_split_point, max_num