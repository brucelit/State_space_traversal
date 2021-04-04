import copy
import heapq
import os
os.environ["PATH"] += os.pathsep + 'E:/Program Files/Graphviz/bin/'
import numpy as np

from pm4py.objects.petri import align_utils as utils
from astar_implementation import heuristic, initialization
from func_timeout import func_set_timeout


@func_set_timeout(500)
def astar_with_split(sync_net, sync_im, sync_fm, aux_dict):

    # initialise closed set, open set, heuristic set
    closed_set = set()
    heuristic_set = set()
    open_set = []
    heapq.heapify(open_set)
    order = 1
    # initialise cost so far function as a dictionary
    dict_g = {}

    # initialise initial set
    ini_state = init_state(sync_im, aux_dict['split_lst'], aux_dict['ini_vec'], aux_dict['fin_vec'],
                           aux_dict['cost_vec'], aux_dict['incidence_matrix'], aux_dict['consumption_matrix'],
                           aux_dict['x_0'], aux_dict['t_index'])
    dict_g[ini_state.marking_tuple] = 0
    heapq.heappush(open_set, ini_state)

    # max events explained
    max_events = max(aux_dict['split_lst'].values())
    split_point = None
    temp_split_lst = copy.deepcopy(aux_dict['split_lst'])

    # while not all states visited
    while len(open_set) > 0:
        aux_dict['visited'] += 1

        # get the most promising marking
        curr = heapq.heappop(open_set)
        curr_vec = initialization.encode_marking(curr.marking, aux_dict['p_index'])

        # final marking reached
        if curr_vec == aux_dict['fin_vec']:
            result = print_result(curr, aux_dict)
            return result

        # heuristic of marking is not exact
        if curr.marking_tuple in heuristic_set:

            # Check if s is not a splitpoint in K
            if split_point not in aux_dict['split_lst']:
                if split_point not in temp_split_lst:
                    aux_dict['split_lst'] = update_split_lst(aux_dict, split_point, curr, max_events)
                    ini_h, ini_parikh_vector, ini_status = heuristic.compute_ini_heuristic(aux_dict['ini_vec'], aux_dict['fin_vec'], aux_dict['cost_vec'],
                                                                                       aux_dict['incidence_matrix'], aux_dict['consumption_matrix'], aux_dict['split_lst'],
                                                                                       aux_dict['x_0'], aux_dict['t_index'])

                # if we do not need to restart
                    if ini_status == "Infeasible":
                        aux_dict['block'] += 1
                        temp_split_lst[split_point] = 1
                        del aux_dict['split_lst'][split_point]
                # if we need to restart
                    else:
                        if np.array_equal(ini_parikh_vector, ini_state.parikh_vector):
                            temp_split_lst[split_point] = 1
                            del aux_dict['split_lst'][split_point]
                            aux_dict['block'] += 1
                        else:
                            aux_dict['restart'] += 1
                            aux_dict['state'] = State(0, ini_h, 0, ini_h, sync_im, tuple(aux_dict['ini_vec']), None, None, [], None,
                                  ini_parikh_vector, 0, 0)
                            return astar_with_split_check(sync_net, sync_im, sync_fm, aux_dict)

            # compute the true estimate heuristic
            new_heuristic, new_parikh_vector = heuristic.compute_exact_heuristic(curr_vec, aux_dict['fin_vec'],
                                                                             aux_dict['incidence_matrix'],
                                                                             aux_dict['cost_vec'])
            aux_dict['recalculation'] = aux_dict['recalculation'] + 1

            # remove marking from heuristic set
            heuristic_set.remove(curr.marking_tuple)
            old_h = curr.h
            curr.not_trust = 0
            curr.parikh_vector = new_parikh_vector
            curr.h = new_heuristic
            if new_heuristic > old_h:
                curr.f = curr.g + new_heuristic
            heapq.heappush(open_set, curr)
            if new_heuristic > old_h:
                continue
        # add marking to the closed set
        closed_set.add(curr.marking_tuple)

        # keep track of the maximum number of events explained, and the x_0 for heuristic computation
        new_max_events = get_max_events(curr)

        if new_max_events > max_events and curr.last_sync != None:
            max_events = new_max_events
            split_point = curr.last_sync
            max_index = aux_dict['t_index'][curr.last_sync]

        # compute enabled transitions and apply model move restriction
        enabled_trans = compute_enabled_transition(curr)

        # For each relevant transition enabled in current marking
        for t in enabled_trans:
            new_marking = utils.add_markings(curr.marking, t.add_marking)
            new_tuple = tuple(initialization.encode_marking(new_marking, aux_dict['p_index']))

            if new_tuple not in closed_set:

                # scenario 1: reach a marking not visited
                if new_tuple not in dict_g:
                    not_trust, h, parikh_vector, g, last_sync  = \
                        visit_new_state(curr, aux_dict['cost_vec'], t, aux_dict['t_index'])
                    f = g + h
                    new_state = State(not_trust, f, g, h, new_marking, new_tuple, curr, t,
                                      last_sync, parikh_vector, order)
                    order += 1
                    dict_g[new_tuple] = g
                    # create new state and add it to open set
                    if not_trust == 1:
                        heuristic_set.add(new_state.marking_tuple)
                    heapq.heappush(open_set, new_state)
                    heapq.heapify(open_set)

                # scenario 2: reach a marking visited
                else:
                    # found shorter path
                    if curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]] < dict_g[new_tuple]:
                        not_trust, new_heuristic, new_parikh_vector, g, last_sync = \
                            visit_new_state(curr, aux_dict['cost_vec'], t, aux_dict['t_index'])
                        dict_g[new_tuple] = curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]]
                        for i in open_set:
                            if i.marking_tuple == new_tuple:
                                i.g = curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]]
                                i.pre_transition = t
                                i.parikh_vecotr = new_parikh_vector
                                i.pre_state = curr
                                i.last_sync = last_sync
                                if not_trust == 1:
                                    i.f = g + max(0, curr.h - aux_dict['cost_vec'][aux_dict['t_index'][t]])
                                    i.h = new_heuristic
                                    if new_state.marking_tuple not in heuristic_set:
                                        heuristic_set.add(new_state.marking_tuple)
                                else:
                                    i.f = g + new_heuristic
                                    i.h = new_heuristic
                                    if new_state.marking_tuple in heuristic_set:
                                        heuristic_set.remove(new_state.marking_tuple)
                        heapq.heapify(open_set)
                        aux_dict['traversed'] += 1



def init_state(sync_im, split_lst, ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, x_0, t_index):
    ini_h, ini_parikh_vector, status = heuristic.compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix,
                                                               consumption_matrix, split_lst, x_0, t_index)
    ini_tuple = tuple(ini_vec)
    ini_f = ini_h
    not_trust = 0
    return State(not_trust, ini_f, 0, ini_h, sync_im, ini_tuple, None, None, None, ini_parikh_vector, 0)


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
    def __init__(self, not_trust, f, g, h, marking, marking_tuple, pre_state, pre_transition, last_sync,
                 parikh_vector, order):
        self.not_trust = not_trust
        self.f = f
        self.g = g
        self.h = h
        self.pre_transition = pre_transition
        self.marking = marking
        self.marking_tuple = marking_tuple
        self.parikh_vector = parikh_vector
        self.pre_state = pre_state
        self.last_sync = last_sync
        self.order = order

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        if self.not_trust != other.not_trust:
            return not self.not_trust
        max_event1 = get_max_events(self)
        max_event2 = get_max_events(other)
        if max_event1 != max_event2:
            return max_event1 > max_event2
        if self.g > other.g:
            return True
        elif self.g < other.g:
            return False
        path1 = get_path_length(self)
        path2 = get_path_length(other)
        if path1 != path2:
            return path1 > path2
        return self.order < other.order


def get_path_length(marking):
    if marking.pre_state == None:
        return 0
    else:
        return 1 + get_path_length(marking.pre_state)


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


def visit_new_state(curr, cost_vec, t, t_index):
    # compute heuristic for new marking
    new_h_score, new_parikh_vector, not_trust = heuristic.compute_estimate_heuristic(curr.h, curr.parikh_vector,
                                                                                     t_index[t], cost_vec)
    # computer last sync transition for new marking
    if t is not None and (t.label[0] == t.label[1]):
        last_sync = t
    else:
        last_sync = curr.last_sync

    # compute cost so far
    g = curr.g + cost_vec[t_index[t]]
    return not_trust, new_h_score, new_parikh_vector, g, last_sync


def update_split_lst(aux_dict, split_point, marking, max_events):
    if len(aux_dict['split_lst']) == 1:
        aux_dict['x_0'] = update_x0(marking, aux_dict['t_index'])

    # add split point to split list
    aux_dict['split_lst'][split_point] = max_events
    return aux_dict['split_lst']


def update_x0(marking, t_index):
    x_0 = [0 for i in t_index]
    max_path = get_path(marking, t_index, [])
    for i in max_path:
        x_0[i] = 1
    return x_0

def get_path(marking, t_index, max_path):
    if marking.pre_state == None:
        return max_path
    else:
        max_path.append(t_index[marking.pre_transition])
        return get_path(marking.pre_state,t_index, max_path)


@func_set_timeout(500)
def astar_with_split_check(sync_net, sync_im, sync_fm, aux_dict):
    # initialise closed set, open set, heuristic set
    closed_set = set()
    heuristic_set = set()
    open_set = []
    heapq.heapify(open_set)

    # initialise cost so far function as a dictionary
    dict_g = {}

    # initialise initial set
    ini_state = aux_dict['state']
    order = 1
    dict_g[ini_state.marking_tuple] = 0
    heapq.heappush(open_set, ini_state)

    # max events explained
    max_events = max(aux_dict['split_lst'].values())
    split_point = None
    temp_split_lst = copy.deepcopy(aux_dict['split_lst'])

    # while not all states visited
    while len(open_set) > 0:
        aux_dict['visited'] += 1

        # get the most promising marking
        curr = heapq.heappop(open_set)
        curr_vec = initialization.encode_marking(curr.marking, aux_dict['p_index'])
        if curr_vec == aux_dict['fin_vec']:
            result = print_result(curr, aux_dict)
            return result

        # heuristic of marking is not exact
        if curr.marking_tuple in heuristic_set:
            # Check if s is not a splitpoint in K
            if split_point not in aux_dict['split_lst']:
                if split_point not in temp_split_lst:
                    aux_dict['split_lst'] =  update_split_lst(aux_dict, split_point, curr, max_events)
                    ini_h, ini_parikh_vector, ini_status = heuristic.compute_ini_heuristic(aux_dict['ini_vec'],
                                                                                           aux_dict['fin_vec'],
                                                                                           aux_dict['cost_vec'],
                                                                                           aux_dict['incidence_matrix'],
                                                                                           aux_dict['consumption_matrix'],
                                                                                           aux_dict['split_lst'],
                                                                                           aux_dict['x_0'],
                                                                                           aux_dict['t_index'])
                    if ini_status == "Infeasible":
                        temp_split_lst[split_point] = 1
                        del aux_dict['split_lst'][split_point]
                        aux_dict['block'] += 1
                    else:
                        if np.array_equal(ini_parikh_vector, ini_state.parikh_vector):
                            temp_split_lst[split_point] = 1
                            del aux_dict['split_lst'][split_point]
                            aux_dict['block'] += 1
                        else:
                            aux_dict['restart'] += 1
                            aux_dict['state'] = State(0, ini_h, 0, ini_h, sync_im, tuple(aux_dict['ini_vec']), None, None,
                                                      [], None, ini_parikh_vector, 0, 0)
                            return astar_with_split_check(sync_net, sync_im, sync_fm, aux_dict)

            # compute the true estimate heuristic
            new_heuristic, new_parikh_vector = heuristic.compute_exact_heuristic(curr_vec, aux_dict['fin_vec'],
                                                                                 aux_dict['incidence_matrix'],
                                                                                 aux_dict['cost_vec'])
            aux_dict['recalculation'] = aux_dict['recalculation'] + 1

            # remove marking from heuristic set
            heuristic_set.remove(curr.marking_tuple)
            old_h = curr.h
            curr.not_trust = 0
            curr.parikh_vector = new_parikh_vector
            curr.h = new_heuristic
            if new_heuristic > old_h:
                curr.f = curr.g + new_heuristic
            heapq.heappush(open_set, curr)
            if new_heuristic > old_h:
                continue

        # add marking to the closed set
        closed_set.add(curr.marking_tuple)

        # keep track of the maximum number of events explained, and the x_0 for heuristic computation
        new_max_events = get_max_events(curr)

        if new_max_events > max_events and curr.last_sync != None:
            max_events = new_max_events
            split_point = curr.last_sync
            max_index = aux_dict['t_index'][curr.last_sync]


        # compute enabled transitions and apply model move restriction
        enabled_trans = compute_enabled_transition(curr)

        # For each relevant transition enabled in current marking
        for t in enabled_trans:
            new_marking = utils.add_markings(curr.marking, t.add_marking)
            new_tuple = tuple(initialization.encode_marking(new_marking, aux_dict['p_index']))

            if new_tuple not in closed_set:
                # scenario 1: reach a marking not visited
                if new_tuple not in dict_g:
                    not_trust, h, parikh_vector, g, last_sync = \
                        visit_new_state(curr, aux_dict['cost_vec'], t, aux_dict['t_index'])
                    # return not_trust, new_h_score, new_parikh_vector, g, t, last_sync
                    f = g + h
                    new_state = State(not_trust, f, g, h, new_marking, new_tuple, curr, t,
                                      last_sync, parikh_vector, order)
                    order += 1
                    dict_g[new_tuple] = g
                    # create new state and add it to open set
                    if not_trust == 1:
                        heuristic_set.add(new_state.marking_tuple)
                    heapq.heappush(open_set, new_state)
                    heapq.heapify(open_set)

                # scenario 2: reach a marking visited
                else:
                    # found shorter path
                    if curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]] < dict_g[new_tuple]:
                        not_trust, new_heuristic, new_parikh_vector, g, last_sync = \
                            visit_new_state(curr, aux_dict['cost_vec'], t, aux_dict['t_index'])
                        dict_g[new_tuple] = curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]]

                        for i in open_set:
                            if i.marking_tuple == new_tuple:
                                i.g = curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]]
                                i.pre_transition = t
                                i.parikh_vecotr = new_parikh_vector
                                i.pre_state = curr
                                i.last_sync = last_sync
                                if not_trust == 1:
                                    i.f = g + max(0, curr.h - aux_dict['cost_vec'][aux_dict['t_index'][t]])
                                    i.h = new_heuristic
                                    if new_state.marking_tuple not in heuristic_set:
                                        heuristic_set.add(new_state.marking_tuple)
                                else:
                                    i.f = g + new_heuristic
                                    i.h = new_heuristic
                                    if new_state.marking_tuple in heuristic_set:
                                        heuristic_set.remove(new_state.marking_tuple)
                        heapq.heapify(open_set)
                        aux_dict['traversed'] += 1
#
def get_max_events(marking):
    if marking.pre_transition == None:
        return 0
    if marking.pre_transition.label[0] == marking.pre_transition.label[1]:
        return get_max_events(marking.pre_state) + 1
    else:
        return get_max_events(marking.pre_state)