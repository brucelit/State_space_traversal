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

from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import func_timeout

@func_set_timeout(500)
def astar_with_split(sync_net, sync_im, sync_fm, aux_dict):
    """
    ------------
    # Line 1-10
    ------------
     """

    closed_set = set()  # init close set
    heuristic_set = set()  # init estimated heuristic set

    #  initial state
    ini_state = init_state(sync_im, aux_dict['split_lst'], aux_dict['ini_vec'], aux_dict['fin_vec'], aux_dict['cost_vec'],
                           aux_dict['incidence_matrix'], aux_dict['consumption_matrix'], aux_dict['x_0'],
                           aux_dict['t_index'])
    open_set = []
    heapq.heapify(open_set)
    heapq.heappush(open_set, ini_state)

    # init the number of states explained
    new_split_point = None
    max_num = -1
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
    # print("current order: ",aux_dict['order'])
    # print("\nbefore checking:", )
    # for i in aux_dict['state_to_check']:
    #     print(i.marking, "not trust:", i.not_trust)
    # print("after checking:")
    # changed_state, valid_path = check_state(aux_dict['state_to_check'], ini_state, aux_dict['sync_trans'])
    # for i in changed_state:
    #     # print("changed state", i.marking, i.not_trust)
    #     heapq.heappush(open_set, i)
    # # print("current open set:")


    '''
    ---------------heapq.heappush(open_set, new_state)---
    # Line 10-30  
    ------------------
    '''
    while len(open_set) > 0:
        # for i in open_set:
        #     print(i.marking, "f:", i.f, "not trust:", i.not_trust, "g:", i.g, "h:", i.h)
        aux_dict['visited'] += 1
        aux_dict['order'] += 1
        # print("\norder:", aux_dict['order'])

        # print("open set:")
        heapq.heapify(open_set)
        # print("open set: ")
        # for i in open_set:
        #     print(i.marking, "f:", i.f, "not trust:", i.not_trust, "g:", i.g, "h:", i.h)
        # print("closed set: ")
        # for i in closed_set:
        #     print(i)
        heapq.heapify(open_set)
        #  favouring markings for which the exact heuristic is known.
        curr = heapq.heappop(open_set)
        # print("check point", curr.parikh_vector)

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
            result = print_result(curr, aux_dict)
            # gviz.graph_attr['label'] = "\nNumber of states visited: " + str(aux_dict['visited']) + \
            #                            "\nNumber of states queued: " + str(aux_dict['queued']) + \
            #                            "\nNumber of edges traversed: " + str(aux_dict['traversed']) + \
            #                            "\nsplit list:" + str(split_lst.keys())
            # ts_visualizer.save(gviz, os.path.join("E:/Thesis/img/acegcd", "step" + str(aux_dict['order']) + ".png"))

            return result

        # ----------Line 16 - 30-----------
        if curr.marking_tuple in heuristic_set:
            # print("current not working")
            if new_split_point not in aux_dict['split_lst']:
                # compute the x_0 in the equation
                if len(aux_dict['split_lst']) == 1:
                    aux_dict['x_0'] = compute_x0(max_path, aux_dict['t_index'])
                else:
                    lst1 = list(aux_dict['split_lst'].values())
                    lst1.sort()
                    if max_num < lst1[1]:
                        # print("max num", max_num)
                        aux_dict['x_0'] = compute_x0(max_path, aux_dict['t_index'])
                        # print("x_0 changed")

                aux_dict['split_lst'][new_split_point] = max_num
                return astar_with_split(sync_net, sync_im, sync_fm, aux_dict)

            # for i in open_set:
            #     print(i.marking, "f:", i.f, "not trust:",i.not_trust, "g:", i.g, "h:", i.h, i.not_trust,i.pre_trans_lst)
            new_heuristic, new_parikh_vector = heuristic.compute_exact_heuristic(curr_vec, aux_dict['fin_vec'],
                                                                                 aux_dict['incidence_matrix'],
                                                                                 aux_dict['cost_vec'])
            aux_dict['recalculation'] = aux_dict['recalculation']+1
            # print("recalculation:", curr.h, "new h:", new_heuristic)
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
            if new_heuristic > old_h:
                # requeue the state after recalculating
                # print("重新计算1")
                continue

        # print("加了",curr.last_sync)
        closed_set.add(curr.marking_tuple)
        # keep track of the maximum number of events explained
        # print("current last sync", curr.f,curr.g,curr.h,curr.last_sync,curr.not_trust)

        new_max_num, new_max_path, lst10,index_10 = max_events_explained(curr, aux_dict)
        # print("close set",closed_set)
        # print("计算结果：", index_10, new_max_num, new_max_path, lst10)
        # print("current last sync", curr.last_sync, new_max_num, new_max_path)
        if new_max_num > max_num:
            max_num = new_max_num
            max_path = new_max_path
            new_split_point = curr.last_sync
            # print("the new split point is:", curr.last_sync, aux_dict['sync_index'][curr.last_sync], curr.pre_trans_lst)
            # print("last sync is:", curr.last_sync)
            # print("max num is:", max_num, "max_path:", max_path)
        '''
        -------------
        # Line 31-end  
        -------------
        '''
        # compute enabled transitions and apply model move restriction
        enabled_trans = compute_enabled_transition(curr)

        #  For each relevant transition enabled in current marking
        for t in enabled_trans:
            new_marking = utils.add_markings(curr.marking, t.add_marking)
            new_tuple = tuple(initialization.encode_marking(new_marking, aux_dict['p_index']))

            if new_tuple not in closed_set:
                # scenario 1: reach a marking not visited
                if new_tuple not in dict_g:
                    # 计算新的state的函数, a是什么，t是什么
                    not_trust, new_heuristic, new_parikh_vector, g, t, new_pre_trans_lst, last_sync = \
                        compute_new_state_test(curr, aux_dict['cost_vec'], t, aux_dict['t_index'])
                    #return not_trust, new_h_score, new_parikh_vector, g, t, new_pre_trans_lst, last_sync
                    f = g + new_heuristic
                    new_state = State(not_trust, f, g, new_heuristic, new_marking, new_tuple, curr, t, new_pre_trans_lst,
                                      last_sync, new_parikh_vector)
                    dict_g[new_tuple] = g
                    # create new state and add it to open set
                    heapq.heappush(open_set, new_state)

                # scenario 2: reach a marking visited
                else:
                    # if the g is smaller than before
                    if curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]] < dict_g[new_tuple]:
                        # print("遇到了smaller than before")
                        not_trust, new_heuristic, new_parikh_vector, g, t, new_pre_trans_lst, last_sync = \
                            compute_new_state_test(curr, aux_dict['cost_vec'], t, aux_dict['t_index'])
                        for i in open_set:
                            if i.marking_tuple == new_tuple:
                                # need to update
                                i.g = curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]]
                                i.pre_transition = t
                                i.parikh_vecotr = new_parikh_vector
                                i.pre_trans_lst = new_pre_trans_lst
                                i.pre_state = curr
                                i.last_sync = last_sync
                                if not_trust == 1:
                                    i.f = g + max(0, curr.h - aux_dict['cost_vec'][aux_dict['t_index'][t]])
                                    i.h = new_heuristic
                                else:
                                    i.f = g + new_heuristic
                                    i.h = new_heuristic
                        aux_dict['traversed'] += 1
                        dict_g[new_tuple] = g
                        heapq.heapify(open_set)

        # gviz = visualization.viz_state_change(aux_dict['ts'], curr_state_lst, valid_state_lst, invalid_state_lst,
        #                                       aux_dict['visited'],
        #                                       split_lst, open_set)
        # print("\nNumber of states visited: " + str(aux_dict['visited']) + "\nsplit list:" + str(
        #     split_lst) + "\nNumber of states in open set: " + str(len(open_set)) +
        #       "\nvalid state: ", valid_state_lst, "\ninvalid state", invalid_state_lst)
        # ts_visualizer.save(gviz, os.path.join("E:/Thesis/img/acegcd", "step" + str(aux_dict['order']) + ".png"))


# return the index of the sync_trans, and the path before
def max_events_explained(curr, aux_dict):
    max_path = []
    # get the split point
    lst2 = []
    cost = 10000
    index2 = -1
    # print(curr.pre_trans_lst)
    # print("start")
    # print(curr.pre_trans_lst)
    for lst in curr.pre_trans_lst:
        max_explained_count = -1
        min_cost = 0
        for i in lst:
            min_cost += aux_dict["trans_aux_dict"][i]
        if min_cost <= cost:
            cost = min_cost
            lst2 = copy.deepcopy(lst)
    #         print("possible：", lst2)
    # # print("end")
    if curr.last_sync is not None:
        if aux_dict['sync_index'][curr.last_sync] > max_explained_count:
            lst3 = duplicates(lst2, aux_dict['t_index'][curr.last_sync])
            # print("lst3", lst3)
            max_explained_count = aux_dict['sync_index'][curr.last_sync]
            index1 = lst3[-1]
            index2 = aux_dict['t_index'][curr.last_sync]
            max_path = lst2[0:index1]
            # print("max exalained", max_explained_count)
            # print("the max path is:", max_path)
            # print("the prev trans is:", lst2)
    return max_explained_count, max_path,lst2, index2


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


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


def init_state(sync_im, split_lst, ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, x_0, t_index,
               REUSE=False):

    ini_h, ini_parikh_vector = heuristic.compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix,
                                                               consumption_matrix, split_lst, x_0, t_index)

    ini_tuple = tuple(ini_vec)
    ini_f = ini_h
    pre_trans_lst = [[]]
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


def print_result(state, aux_dict):
    result = reconstruct_alignment(state, aux_dict, False)
    return result


def reconstruct_alignment(state, aux_dict, ret_tuple_as_trans_desc=False):
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
    result = {"alignment": alignment, "cost": state.g, "visited_states": aux_dict['visited'],
              "queued_states": aux_dict['queued'], "traversed_arcs": aux_dict['traversed'],
              "split": len(aux_dict['split_lst'])-1, 'block_restart': aux_dict['block'], 'h_recalculation': aux_dict['recalculation']}
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
        else:
            return self.g < other.g

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
