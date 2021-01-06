import copy
import heapq
import os
import numpy as np
import pandas as pd


from pm4py.objects.petri import align_utils as utils
from pm4py.visualization.transition_system import visualizer as ts_visualizer

from astar_implementation import heuristic
from astar_implementation import construction as utilities
from astar_implementation import visualization, initialization

ret_tuple_as_trans_desc = False


# add a dynamic set called far_set, only keeps the node not
def astar_with_split(sync_net, sync_im, sync_fm, aux_dict, split_lst):
    """
    ------------
    # Line 1-10
    ------------
     """

    closed_set = set()  # init close set
    heuristic_set = set()  # init estimated heuristic set

    #  initial state
    ini_state = init_state(sync_im, split_lst, aux_dict['ini_vec'], aux_dict['fin_vec'], aux_dict['cost_vec'],
                           aux_dict['incidence_matrix'], aux_dict['consumption_matrix'], aux_dict['x_0'],
                           aux_dict['t_index'], sync_net, aux_dict, sync_im)
    open_set = []
    heapq.heapify(open_set)
    heapq.heappush(open_set, ini_state)

    # init the number of states explained
    new_split_point = None
    max_num = 0
    max_path = []

    # matrices for measurement
    valid_state_lst = set()
    invalid_state_lst = set()

    # dict_g来表示他是否被访问过, key是marking_tuple
    dict_g = {ini_state.marking_tuple: 0}

    # use state_path to represent all the possible path leading to current state
    state_path = {ini_state.marking_tuple: [[]]}
    lst2 = marking_to_list(ini_state.marking, aux_dict['place_map'])
    valid_state_lst.add(lst2)
    print("current order: ",aux_dict['order'])
    print("\nbefore checking:", )
    for i in aux_dict['state_to_check']:
        print(i.marking, "not trust:", i.not_trust)
    print("after checking:")
    changed_state, valid_path = check_state(aux_dict['state_to_check'], ini_state, aux_dict['sync_trans'])
    for i in changed_state:
        print("changed state", i.marking, i.not_trust)
        heapq.heappush(open_set, i)
    print("current open set:")
    for i in open_set:
        print(i.marking, "f:", i.f, "not trust:", i.not_trust, "g:", i.g, "h:", i.h)

    '''
    ---------------heapq.heappush(open_set, new_state)---
    # Line 10-30  
    ------------------
    '''
    while len(open_set) > 0:
        aux_dict['visited'] += 1
        aux_dict['order'] += 1
        print("\norder:", aux_dict['order'])

        # print("open set:")
        heapq.heapify(open_set)
        for i in open_set:
            print(i.marking,i.f,i.not_trust,i.g,i.h)
        # for i in open_set:
        #     print(i.marking, "f:", i.f, "not trust:", i.not_trust, "g:", i.g, "h:", i.h)
        heapq.heapify(open_set)
        #  favouring markings for which the exact heuristic is known.
        curr = heapq.heappop(open_set)
        # print("order:", aux_dict['order'])
        # print("current marking:", curr.marking, curr.pre_trans_lst)
        # tranform places in current state to list in form p1,p2,p3…

        lst2 = marking_to_list(curr.marking, aux_dict['place_map'])

        if curr.not_trust == 0 and lst2 not in valid_state_lst:
            valid_state_lst.add(lst2)
            if lst2 in invalid_state_lst:
                invalid_state_lst.remove(lst2)
        if curr.not_trust == 1 and lst2 not in valid_state_lst and lst2 not in invalid_state_lst:
            invalid_state_lst.add(lst2)
        # visualize and save change
        # gviz = visualization.viz_state_change(aux_dict['ts'], curr_state_lst, valid_state_lst, invalid_state_lst,
        #                                       aux_dict['visited'], split_lst, open_set)
        # ts_visualizer.save(gviz, os.path.join("E:/Thesis/img/acegcd", "step" + str(aux_dict['order']) + ".png"))

        curr_vec = initialization.encode_marking(curr.marking, aux_dict['p_index'])
        if curr_vec == aux_dict['fin_vec']:
            result = print_result(curr, aux_dict['visited'], aux_dict['queued'], aux_dict['traversed'], split_lst)
            # gviz.graph_attr['label'] = "\nNumber of states visited: " + str(aux_dict['visited']) + \
            #                            "\nNumber of states queued: " + str(aux_dict['queued']) + \
            #                            "\nNumber of edges traversed: " + str(aux_dict['traversed']) + \
            #                            "\nsplit list:" + str(split_lst.keys())
            # ts_visualizer.save(gviz, os.path.join("E:/Thesis/img/acegcd", "step" + str(aux_dict['order']) + ".png"))

            return result

        if curr.marking_tuple in heuristic_set:
            # print("find in heuristic set")
            # print("solution vector",curr.parikh_vector)
            if new_split_point not in split_lst:
                print("检查：", new_split_point, "max path", max_path, "current split lst:", split_lst)
                if len(split_lst) == 1:
                    aux_dict['x_0'] = compute_x0(max_path, aux_dict['t_index'])
                else:
                    lst1 = list(split_lst.values())
                    lst1.sort()
                    if max_num < lst1[1]:
                        aux_dict['x_0'] = compute_x0(max_path, aux_dict['t_index'])
                        print("x_0 changed")
                split_lst[new_split_point] = max_num

                new_ini_h, new_ini_parikh_vector = heuristic.compute_ini_heuristic(aux_dict['ini_vec'], aux_dict['fin_vec'],
                                                                                   aux_dict['cost_vec'], aux_dict['incidence_matrix'],
                                                    aux_dict['consumption_matrix'], split_lst, aux_dict['x_0'], aux_dict['t_index'],
                                                                                   sync_net, aux_dict, curr.marking)
                print("h for initial and current h", ini_state.h, new_ini_h)
                print("parikh vector:", new_ini_parikh_vector, ini_state.parikh_vector)
                if new_ini_h == ini_state.h and np.equal(new_ini_parikh_vector.all(), ini_state.parikh_vector.all()):
                    print("show up 1！")
                    pass
                else:
                    print("show up 2")
                    print("current max", max_num)
                    heapq.heappush(open_set, curr)
                    aux_dict['state_to_check'] = []
                    for i in open_set:
                        if i.not_trust == 1:
                            aux_dict['state_to_check'].append(i)
                    return astar_with_split(sync_net, sync_im, sync_fm, aux_dict, split_lst)

            new_heuristic, new_parikh_vector = heuristic.compute_exact_heuristic(curr_vec, aux_dict['fin_vec'],
                                                                                 aux_dict['incidence_matrix'], aux_dict['cost_vec'])
            print("recalculation:", curr.h, "new h:", new_heuristic)
            heuristic_set.remove(curr.marking_tuple)
            old_h = curr.h
            curr.not_trust = 0
            curr.parikh_vector = new_parikh_vector
            curr.h = new_heuristic
            if lst2 not in valid_state_lst:
                valid_state_lst.add(lst2)
                if lst2 in invalid_state_lst:
                    invalid_state_lst.remove(lst2)
            if new_heuristic > old_h:
                curr.f = curr.g + new_heuristic
                # print("new f", curr.f)
                # print("new h", curr.h)
                # print("new not trust", curr.not_trust)
                heapq.heappush(open_set, curr)
                # requeue the state after recalculating
                continue
        closed_set.add(curr.marking_tuple)

        # keep track of the maximum number of events explained
        new_max_num, new_max_path = max_events_explained(curr, aux_dict['sync_map'])
        if new_max_num > max_num:
            max_num = new_max_num
            max_path = new_max_path
            new_split_point = curr.last_sync
            print("last sync:", curr.last_sync)
            print("max num:", max_num, "max_path:", max_path)
        '''
        -------------
        # Line 31-end  
        -------------
        '''
        # compute enabled transitions and apply model move restriction
        enabled_trans = compute_enabled_transition(curr)

        for t in enabled_trans:
            new_marking = utils.add_markings(curr.marking, t.add_marking)
            new_tuple = tuple(initialization.encode_marking(new_marking, aux_dict['p_index']))
            # reach a marking not yet visited
            if new_tuple not in dict_g:
                dict_g[new_tuple] = 10000
            if new_tuple not in closed_set:
                aux_dict['traversed'] += 1
                # create new state
                not_trust, new_heuristic, new_parikh_vector, a, t, new_pre_trans_lst, last_sync = \
                    compute_new_state_test(curr, aux_dict['cost_vec'], t, aux_dict['t_index'])
                if new_tuple not in state_path:
                    state_path[new_tuple] = new_pre_trans_lst
                else:
                    for i in new_pre_trans_lst:
                        state_path[new_tuple].append(i)
                # print("new marking:", new_marking)
                # print("path for new state", state_path[new_tuple])

                if a <= dict_g[new_tuple]:
                    g = a
                    if not_trust == 1:
                        f = g + max(0, curr.h - aux_dict['cost_vec'][aux_dict['t_index'][t]])
                        h = new_heuristic
                    else:
                        f = g + new_heuristic
                        h = new_heuristic
                    new_state = State(not_trust, f, g, h, new_marking, new_tuple, curr, t, new_pre_trans_lst,
                                           last_sync, new_parikh_vector)
                    lst2 = marking_to_list(new_marking, aux_dict['place_map'])
                    if not_trust == 0:
                        if lst2 in invalid_state_lst:
                            # print("yes, remove!!!")
                            invalid_state_lst.remove(lst2)
                            heuristic_set.remove(new_state.marking_tuple)
                        valid_state_lst.add(lst2)
                    else:
                        heuristic_set.add(new_state.marking_tuple)
                        invalid_state_lst.add(lst2)
                        if lst2 in valid_state_lst:
                            heuristic_set.remove(new_state.marking_tuple)
                            invalid_state_lst.remove(lst2)
                    dict_g[new_tuple] = new_state.g
                # whether add this state to open set
                flag = 1
                for i in open_set:
                    if i.marking_tuple == new_tuple:
                        if f < i.f and not not_trust:
                            # print("success remove", i.marking)
                            open_set.remove(i)
                            heapq.heapify(open_set)
                        elif f == i.f and i.not_trust and not not_trust:
                            # print("success remove", i.marking)
                            open_set.remove(i)
                            heapq.heapify(open_set)
                        else:
                            flag = 0
                if flag == 1:
                    heapq.heappush(open_set, new_state)
                    aux_dict['queued'] += 1
        #
        # gviz = visualization.viz_state_change(aux_dict['ts'], curr_state_lst, valid_state_lst, invalid_state_lst,
        #                                       aux_dict['visited'],
        #                                       split_lst, open_set)
        # print("\nNumber of states visited: " + str(aux_dict['visited']) + "\nsplit list:" + str(
        #     split_lst) + "\nNumber of states in open set: " + str(len(open_set)) +
        #       "\nvalid state: ", valid_state_lst, "\ninvalid state", invalid_state_lst)
        # ts_visualizer.save(gviz, os.path.join("E:/Thesis/img/acegcd", "step" + str(aux_dict['order']) + ".png"))


# return the index of the sync_trans, and the path before
def max_events_explained(curr, sync_map):
    max_path = []
    # get the split point
    for lst in curr.pre_trans_lst:
        max_explained_count = -1
        lst1 = list(sync_map.keys())
        last_sync = None
        for i in lst:
            if i in lst1:
                last_sync = i
        if last_sync is not None:
            if sync_map[last_sync] > max_explained_count:
                max_explained_count = sync_map[last_sync]
                max_path = lst[0:max_explained_count-1]
    return max_explained_count, max_path


def compute_x0(trans_lst, t_index):
    x_0 = [0 for i in t_index]
    for i in trans_lst:
        x_0[i] = 1
    return x_0


def check_state(state_to_check, ini_state, sync_trans):
    changed_state = []
    valid_path = []
    for i in range(len(state_to_check)):
        state = state_to_check[i]
        for path in state.pre_trans_lst:
            temp_not_trust = 0
            for trans in path:
                if ini_state.parikh_vector[trans] < 1:
                    temp_not_trust = 1
            if temp_not_trust == 0:
                state.not_trust = 0
                h = ini_state.h
                g = ini_state.g
                for j in path:
                    if j not in sync_trans:
                        h -= 1
                        g += 1
                state.h = h
                state.g = g
                state.f = h + g
                changed_state.append(state)
                valid_path.append(path)
    return changed_state, valid_path


def init_state(sync_im, split_lst, ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, x_0, t_index, sync_net, aux_dict, marking):
    ini_h, ini_parikh_vector = heuristic.compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix,
                                                               consumption_matrix, split_lst, x_0, t_index,
                                                               sync_net, aux_dict, marking)
    print("heuristic computed: ", ini_h)
    ini_tuple = tuple(ini_vec)
    ini_f = ini_h
    pre_trans_lst = [[]]
    # print("ini_h:", ini_h)
    # print("solution vector:", ini_parikh_vector)
    # not_trust equal to 0 means the solution vector is known.
    not_trust = 0
    return State(not_trust, ini_f, 0, ini_h, sync_im, ini_tuple, None, None, pre_trans_lst, None, ini_parikh_vector)


def compute_new_state_test(curr, cost_vec, t, t_index):
    # for i in curr.pre_trans_lst:
    new_pre_trans_lst = copy.deepcopy(curr.pre_trans_lst)
    for i in new_pre_trans_lst:
        index_to_add = t_index[t]
        i.append(index_to_add)
    new_h_score, new_parikh_vector, not_trust = heuristic.compute_estimate_heuristic(curr.h, curr.parikh_vector,
                                                                                     t_index[t], cost_vec)

    if t is not None and (t.label[0] == t.label[1]):
        last_sync = t
    else:
        last_sync = curr.last_sync
    # compute cost so far
    a = curr.g + cost_vec[t_index[t]]
    # print("previous trans for new state", new_pre_trans_lst)
    return not_trust, new_h_score, new_parikh_vector, a, t, new_pre_trans_lst, last_sync


def marking_to_list(marking, place_map):
    element_valid_state = []
    for p in sorted(marking, key=lambda k: k.name[1]):
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
    result = reconstruct_alignment(state, visited, queued, traversed, split_lst, False)
    # print("Optimal alignment:", result["alignment"], "\nCost of optimal alignment:",
    #       result["cost"], "\nNumber of states visited:", result["visited_states"],
    #       "\nNumber of split: " + str(len(split_lst) - 1) + "\nTransition in split set: " + \
    #       str(split_lst) + "\nF-score for final state: " + \
    #       str(state.f) + "\nNumber of states visited: " + str(visited) + \
    #       "\nNumber of states queued: " + str(queued) + \
    #       "\nNumber of edges traversed: " + str(traversed))
    return result


def reconstruct_alignment(state, visited, queued, traversed, split, ret_tuple_as_trans_desc=False):
    parent = state.pre_state
    if ret_tuple_as_trans_desc:
        alignment = [(state.pre_transition.name, state.pre_transition.label)]
        while parent.pre_state is not None:
            alignment = [(parent.pre_transition.name, parent.pre_transition.label)] + alignment
            parent = parent.pre_state
    else:
        alignment = [state.pre_transition.label]
        while parent.pre_state is not None:
            alignment = [parent.pre_transition.label] + alignment
            parent = parent.pre_state
    result = {"trace": "", "cost": state.g, "visited_states": visited, "queued_states": queued,
              "traversed_edges": traversed, "split": len(split)-1, "alignment": alignment, }
    return result


class State:
    def __init__(self, not_trust, f, g, h, marking, marking_tuple, pre_state, pre_transition, pre_trans_lst, last_sync,
                 parikh_vector):
        self.not_trust = not_trust
        self.f = f
        self.g = g
        self.h = h
        self.pre_transition = pre_transition
        self.marking = marking
        self.marking_tuple = marking_tuple
        self.parikh_vector = parikh_vector
        self.pre_trans_lst = pre_trans_lst
        self.pre_state = pre_state
        self.last_sync = last_sync


    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif other.f < self.f:
            return False
        elif not self.not_trust and other.not_trust:
            return True
        elif self.not_trust and not other.not_trust:
            return False
        # elif self.h < other.h:
        #     return True
        # elif self.h > other.h:
        #     return False
        else:
            return self.g > other.g
        # else:
        #     return self.h < other.h
    # def __lt__(self, other):
    #     return (self.f, self.not_trust, other.g) < (other.f, other.not_trust, self.g)
    #
    # def __gt__(self, other):
    #     return (self.f, other.not_trust, other.g) > (other.f, other.not_trust, self.g)
    #
    # def __eq__(self, other):
    #     return (self.f, self.not_trust, self.g) == (other.f, other.not_trust, other.g)
    # def __lt__(self, other):
    #     return (self.not_trust, self.f, other.g) < (other.not_trust, other.f, self.g)
    #
    # def __gt__(self, other):
    #     return (other.not_trust, self.f, other.g) > (other.not_trust, other.f, self.g)
    #
    # def __eq__(self, other):
    #     return (self.f, self.not_trust, self.g) == (other.f, other.not_trust, other.g)

