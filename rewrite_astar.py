import copy
import heapq

from pm4py.objects.petri import align_utils as utils
from astar_implementation import heuristic
from astar_implementation import visualization, initialization
from func_timeout import func_set_timeout


@func_set_timeout(500)
def astar_with_split(sync_net, sync_im, sync_fm, aux_dict):

    # initialise closed set, open set, heuristic set
    closed_set = set()
    heuristic_set = set()
    open_set = []
    heapq.heapify(open_set)

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
    max_path = []

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
                aux_dict['split_lst'] = update_split_lst(aux_dict, split_point, max_events, max_path)
                # print(split_point, max_events)
                return astar_with_split(sync_net, sync_im, sync_fm, aux_dict)

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
        new_max_events = curr.max_events

        if new_max_events > max_events and curr.last_sync != None:
            # print(curr.last_sync, max_events, new_max_events)
            max_events = new_max_events
            split_point = curr.last_sync
            max_index = aux_dict['t_index'][curr.last_sync]
            max_index1 = curr.pre_trans_lst.index(max_index)
            max_path = copy.deepcopy(curr.pre_trans_lst[0:max_index1])

        # compute enabled transitions and apply model move restriction
        enabled_trans = compute_enabled_transition(curr)

        # For each relevant transition enabled in current marking
        for t in enabled_trans:
            new_marking = utils.add_markings(curr.marking, t.add_marking)
            new_tuple = tuple(initialization.encode_marking(new_marking, aux_dict['p_index']))

            if new_tuple not in closed_set:
                if t.label[0] == t.label[1]:
                    new_max_events_explained = curr.max_events + 1
                else:
                    new_max_events_explained = curr.max_events

                # scenario 1: reach a marking not visited
                if new_tuple not in dict_g:
                    not_trust, h, parikh_vector, g, pre_trans_lst, last_sync  = \
                        visit_new_state(curr, aux_dict['cost_vec'], t, aux_dict['t_index'])
                    # return not_trust, new_h_score, new_parikh_vector, g, t, new_pre_trans_lst, last_sync
                    f = g + h
                    new_state = State(not_trust, f, g, h, new_marking, new_tuple, curr, t, pre_trans_lst,
                                      last_sync, parikh_vector, new_max_events_explained)
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
                        not_trust, new_heuristic, new_parikh_vector, g, new_pre_trans_lst, last_sync = \
                            visit_new_state(curr, aux_dict['cost_vec'], t, aux_dict['t_index'])
                        dict_g[new_tuple] = curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]]
                        for i in open_set:
                            if i.marking_tuple == new_tuple:
                                # need to update
                                # print("g before", i.g)
                                i.g = curr.g + aux_dict['cost_vec'][aux_dict['t_index'][t]]
                                # print("g after", i.g)
                                # print("pre trans lst", i.pre_trans_lst)
                                i.pre_transition = t
                                i.parikh_vecotr = new_parikh_vector
                                i.pre_trans_lst = new_pre_trans_lst
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
    ini_h, ini_parikh_vector = heuristic.compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix,
                                                               consumption_matrix, split_lst, x_0, t_index)

    ini_tuple = tuple(ini_vec)
    ini_f = ini_h
    pre_trans_lst = []
    not_trust = 0
    return State(not_trust, ini_f, 0, ini_h, sync_im, ini_tuple, None, None, pre_trans_lst, None, ini_parikh_vector, 0)


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
                 parikh_vector, max_events):
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
        self.max_events = max_events

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
            return self.g > other.g


def compute_enabled_transition(state):
    possible_enabling_transitions = set()
    for p in state.marking:
        for t in p.ass_trans:
            possible_enabling_transitions.add(t)
    enabled_trans = [t for t in possible_enabling_transitions if t.sub_marking <= state.marking]
    # violated_trans = []
    # for t in enabled_trans:
    #     if state.pre_transition is None:
    #         break
    #     if state.pre_transition.label[0] == ">>" and t.label[1] == ">>":
    #         violated_trans.append(t)
    # for t in violated_trans:
    #     enabled_trans.remove(t)
    return sorted(enabled_trans, key=lambda k: k.label)


def visit_new_state(curr, cost_vec, t, t_index):
    # compute previous transition list
    new_pre_trans_lst = copy.deepcopy(curr.pre_trans_lst)
    new_pre_trans_lst.append(t_index[t])

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
    return not_trust, new_h_score, new_parikh_vector, g, new_pre_trans_lst, last_sync


def update_split_lst(aux_dict, split_point, max_num, max_path):
    # compute x_0
    if len(aux_dict['split_lst']) == 1:
        aux_dict['x_0'] = update_x0(max_path, aux_dict['t_index'])

    # get the min value in split list
    all_values = aux_dict['split_lst'].values()
    min_expl = min(all_values)

    # if max_num < min, then we need to update the x_0
    if max_num < min_expl:
        print("需要更新x_0了")
        aux_dict['x_0'] = update_x0(max_path, aux_dict['t_index'])

    # add split point to split list
    aux_dict['split_lst'][split_point] = max_num
    # print("split list", aux_dict['split_lst'])
    return aux_dict['split_lst']

def update_x0(max_path, t_index):
    x_0 = [0 for i in t_index]
    for i in max_path:
        x_0[i] = 1
    return x_0