import heapq
import sys
import timeit
from enum import Enum
import re
import numpy as np
from copy import deepcopy, copy
from pm4py.objects.petri import align_utils as utils
from pm4py.objects.petri.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri.utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from heuristic import compute_ini_heuristic, compute_exact_heuristic


class Parameters(Enum):
    PARAM_TRACE_COST_FUNCTION = 'trace_cost_function'
    PARAM_MODEL_COST_FUNCTION = 'model_cost_function'
    PARAM_SYNC_COST_FUNCTION = 'sync_cost_function'
    PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE = 'ret_tuple_as_trans_desc'
    PARAM_TRACE_NET_COSTS = "trace_net_costs"
    TRACE_NET_CONSTR_FUNCTION = "trace_net_constr_function"
    TRACE_NET_COST_AWARE_CONSTR_FUNCTION = "trace_net_cost_aware_constr_function"
    PARAM_MAX_ALIGN_TIME_TRACE = "max_align_time_trace"
    PARAM_MAX_ALIGN_TIME = "max_align_time"
    PARAMETER_VARIANT_DELIMITER = "variant_delimiter"
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    VARIANTS_IDX = "variants_idx"


PARAM_TRACE_COST_FUNCTION = Parameters.PARAM_TRACE_COST_FUNCTION.value
PARAM_MODEL_COST_FUNCTION = Parameters.PARAM_MODEL_COST_FUNCTION.value
PARAM_SYNC_COST_FUNCTION = Parameters.PARAM_SYNC_COST_FUNCTION.value


def apply(trace, petri_net, initial_marking, final_marking, parameters=None):
    """
    Performs the basic alignment search, given a trace and a net.
    Parameters
    ----------
    trace: :class:`list` input trace, assumed to be a list of events (i.e. the code will use the activity key
    to get the attributes)
    petri_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net to use in the alignment
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the Petri net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the Petri net
    parameters: :class:`dict` (optional) dictionary containing one of the following:
        Parameters.PARAM_TRACE_COST_FUNCTION: :class:`list` (parameter) mapping of each index of the trace to a positive cost value
        Parameters.PARAM_MODEL_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        model cost
        Parameters.PARAM_SYNC_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        synchronous costs
        Parameters.ACTIVITY_KEY: :class:`str` (parameter) key to use to identify the activity described by the events
    Returns
    -------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    if parameters is None:
        parameters = {}

    parameters = copy(parameters)
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY)
    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    trace_net_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_CONSTR_FUNCTION, parameters,
                                                           None)
    trace_net_cost_aware_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_COST_AWARE_CONSTR_FUNCTION,
                                                                      parameters, construct_trace_net_cost_aware)

    if trace_cost_function is None:
        trace_cost_function = list(
            map(lambda e: utils.STD_MODEL_LOG_MOVE_COST, trace))
        parameters[Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

    if model_cost_function is None:
        # reset variables value
        model_cost_function = dict()
        sync_cost_function = dict()
        for t in petri_net.transitions:
            if t.label is not None:
                model_cost_function[t] = utils.STD_MODEL_LOG_MOVE_COST
                sync_cost_function[t] = 0
            else:
                model_cost_function[t] = 1
        parameters[Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
        parameters[Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function
    trace_net, trace_im, trace_fm, parameters[
        Parameters.PARAM_TRACE_NET_COSTS] = trace_net_cost_aware_constr_function(trace,
                                                                                 trace_cost_function,
                                                                                 activity_key=activity_key)
    alignment = apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm,
                                parameters)
    return alignment


def apply_from_variant(variant, petri_net, initial_marking, final_marking, parameters=None):
    """
    Apply the alignments from the specification of a single variant
    Parameters
    -------------
    variant
        Variant (as string delimited by the "variant_delimiter" parameter)
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm (same as 'apply' method, plus 'variant_delimiter' that is , by default)
    Returns
    ------------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    if parameters is None:
        parameters = {}
    trace = variants_util.variant_to_trace(variant, parameters=parameters)

    return apply(trace, petri_net, initial_marking, final_marking, parameters=parameters)


def apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm,
                    parameters=None):
    if parameters is None:
        parameters = {}

    ret_tuple_as_trans_desc = exec_utils.get_param_value(Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE,
                                                         parameters, False)

    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    sync_cost_function = exec_utils.get_param_value(Parameters.PARAM_SYNC_COST_FUNCTION, parameters, None)
    trace_net_costs = exec_utils.get_param_value(Parameters.PARAM_TRACE_NET_COSTS, parameters, None)

    revised_sync = dict()
    for t_trace in trace_net.transitions:
        for t_model in petri_net.transitions:
            if t_trace.label == t_model.label:
                revised_sync[(t_trace, t_model)] = sync_cost_function[t_model]

    sync_prod, sync_initial_marking, sync_final_marking, cost_function = construct_cost_aware(
        trace_net, trace_im, trace_fm, petri_net, initial_marking, final_marking, utils.SKIP,
        trace_net_costs, model_cost_function, revised_sync)
    max_align_time_trace = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                      sys.maxsize)

    decorate_transitions_prepostset(trace_net)
    decorate_places_preset_trans(trace_net)

    trans_empty_preset = set(t for t in trace_net.transitions if len(t.in_arcs) == 0)
    current_marking = trace_im

    trace_lst = []
    while current_marking != trace_fm:
        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)
                    trace_lst.append(t)
        for t in enabled_trans:
            new_marking = utils.add_markings(current_marking, t.add_marking)
        current_marking = new_marking

    return apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function, trace_lst,
                           utils.SKIP, ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                           max_align_time_trace=max_align_time_trace)


def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, trace_lst, skip,
                    ret_tuple_as_trans_desc=False,
                    max_align_time_trace=sys.maxsize):
    decorate_transitions_prepostset(sync_prod)
    decorate_places_preset_trans(sync_prod)

    incidence_matrix = construct(sync_prod)
    split_lst = []
    split_lst.append(-1)
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1
    trace_sync = [None for i in range(0, len(trace_lst))]
    trace_log = [None for i in range(0, len(trace_lst))]
    t_index = incidence_matrix.transitions
    for t in sync_prod.transitions:
        for i in range(len(trace_lst)):
            if trace_lst[i].name == t.name[0] and t.label[1] == ">>":
                trace_log[i] = t_index[t]
            if trace_lst[i].name == t.name[0] and t.label[1] != ">>":
                trace_sync[i] = t_index[t]
    restart = 0
    block_restart = 0
    start_time = timeit.default_timer()
    time_h = 0
    res = search(sync_prod, initial_marking, final_marking, cost_function, skip, split_lst, incidence_matrix, {},
                  restart, block_restart, visited, queued, traversed, lp_solved, trace_sync, trace_log, time_h)
    res['time_sum'] = timeit.default_timer() - start_time
    res['time_diff'] = res['time_sum'] - res['time_h']
    return res


def search(sync_net, ini, fin, cost_function, skip, split_lst, incidence_matrix, init_dict,
           restart, block_restart, visited, queued, traversed, lp_solved, trace_sync, trace_log, time_h, use_init=False):
    ini_vec, fin_vec, cost_vec = vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)
    visited_temp = 0
    closed = set()
    t_index = incidence_matrix.transitions
    p_index = incidence_matrix.places
    set_model_move = []
    for t in t_index:
        if t.label[0] == ">>":
            set_model_move.append(t_index[t])

    if use_init:
        h, x, trustable = init_dict['h'], init_dict['x'], True
    else:
        start_time = timeit.default_timer()
        h, x = compute_exact_heuristic(ini_vec, fin_vec, incidence_matrix.a_matrix, cost_vec)
        time_h += timeit.default_timer() - start_time

    open_set = []
    order = 0
    ini_state = SearchTuple(0 + h, 0, h, ini, None, None, x, True, [], order)
    open_set.append(ini_state)
    heapq.heapify(open_set)
    max_events = -1
    dict_g = {ini: 0}
    init_dict = {}

    #  While not all states visited
    while not len(open_set) == 0:
        # Get the most promising marking
        curr = heapq.heappop(open_set)
        # final marking reached
        if curr.m == fin:
            # print(len(split_lst), "round:", visited_temp)
            return reconstruct_alignment(curr, visited, queued, traversed, lp_solved, restart, len(trace_log), time_h)

        # heuristic of m is not exact
        if not curr.trust:

            # check if s is not already a splitpoint in K
            if max_events not in split_lst:

                # Add s to the maximum events explained to K
                split_lst.append(max_events)
                start_time = timeit.default_timer()
                h, x, trustable = compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix.a_matrix,
                                                        incidence_matrix.b_matrix, split_lst, t_index, p_index,
                                                        trace_sync, trace_log)
                time_h += timeit.default_timer() - start_time
                lp_solved += 1
                init_dict['x'] = x
                init_dict['h'] = h
                restart += 1
                heapq.heappush(open_set, curr)
                return search(sync_net, ini, fin, cost_function, skip, split_lst, incidence_matrix, init_dict,
                              restart, block_restart, visited, queued, traversed, lp_solved, trace_sync,
                              trace_log, time_h, use_init=True)

            # compute the true heuristic
            start_time = timeit.default_timer()
            h, x = compute_exact_heuristic(incidence_matrix.encode_marking(curr.m),
                                           fin_vec,
                                           incidence_matrix.a_matrix,
                                           cost_vec)
            time_h += timeit.default_timer() - start_time

            lp_solved += 1
            if h > curr.h:
                tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True, curr.pre_trans_lst, curr.order)
                heapq.heappush(open_set, tp)
                heapq.heapify(open_set)
                continue

        closed.add(curr.m)
        new_max_events, last_sync = get_max_events(curr)
        if new_max_events > max_events and last_sync is not None and new_max_events not in split_lst:
            max_events = new_max_events

        visited += 1
        visited_temp += 1
        enabled_trans = set()
        for p in curr.m:
            for t in p.ass_trans:
                if t.sub_marking <= curr.m:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans]
        enabled_trans = sorted(sorted(trans_to_visit_with_cost, key=lambda k: k[1]), key=lambda k: k[0].label[0])
        for t, cost in enabled_trans:
            traversed += 1
            new_marking = utils.add_markings(curr.m, t.add_marking)
            if new_marking in closed:
                continue
            if new_marking not in dict_g:
                g = curr.g + cost
                dict_g[new_marking] = g
                queued += 1
                h, x = derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                trustable = trust_solution(x)
                new_f = g + h
                pre_trans = deepcopy(curr.pre_trans_lst)
                pre_trans.append(t_index[t])
                order += 1
                tp = SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable, pre_trans, order)
                heapq.heappush(open_set, tp)
            else:
                if curr.g + cost < dict_g[new_marking]:
                    dict_g[new_marking] = curr.g + cost
                    for i in open_set:
                        if i.m == new_marking:
                            pre_trans = deepcopy(curr.pre_trans_lst)
                            pre_trans.append(t_index[t])
                            i.pre_trans_lst = pre_trans
                            i.g = curr.g + cost
                            queued += 1
                            i.h, i.x = derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                            i.trust = trust_solution(i.x)
                            i.f = i.g + i.h
                            i.t = t
                            i.p = curr
                            i.order = curr.order + 1
                            break
        heapq.heapify(open_set)


class SearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust, pre_trans_lst, order):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        self.pre_trans_lst = pre_trans_lst
        self.order = order

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        if self.trust != other.trust:
            return self.trust
        max_event1, t1 = get_max_events(self)
        max_event2, t2 = get_max_events(other)
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
        if self.order < other.order:
            return True
        else:
            return False

    def __get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.__get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " f=" + str(self.f), ' g=' + str(self.g), " h=" + str(self.h),
                        " path=" + str(self.__get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)


def get_max_events(marking):
    if marking.t is None:
        return 0, None
    if marking.t.label[0] != ">>":
        return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group()) + 1, marking.t
    return get_max_events(marking.p)


def get_max_events2(marking):
    if marking.t is None:
        return 0
    if marking.t.label[0] == marking.t.label[1]:
        return 1 + get_max_events2(marking.p)


def get_path_length(marking):
    if marking.p is None:
        return 0
    else:
        return 1 + get_path_length(marking.p)


def get_pre_events(marking, lst):
    if marking.t is None:
        return lst
    lst.insert(0, marking.t.label)
    return get_pre_events(marking.p, lst)


def get_pre_trans(marking, lst):
    if marking.t is None:
        return lst
    lst.insert(0, marking.t)
    return get_pre_trans(marking.p, lst)


def check_heuristic(state, ini_vec):
    solution_vec = deepcopy(ini_vec)
    for i in state.pre_trans_lst:
        solution_vec[i] -= 1
    for j in solution_vec:
        if j < 0:
            return False
    return True


def reconstruct_alignment(state, visited, queued, traversed, lp_solved, restart, trace_length, time_h, ret_tuple_as_trans_desc=False):
    alignment = list()
    if state.p is not None and state.t is not None:
        parent = state.p
        if ret_tuple_as_trans_desc:
            alignment = [(state.t.name, state.t.label)]
            while parent.p is not None:
                alignment = [(parent.t.name, parent.t.label)] + alignment
                parent = parent.p
        else:
            alignment = [state.t.label]
            while parent.p is not None:
                alignment = [parent.t.label] + alignment
                parent = parent.p
    return {'alignment': alignment,
            'cost': state.g,
            'visited_states': visited,
            'queued_states': queued,
            'traversed_arcs': traversed,
            'lp_solved': lp_solved,
            'restart': restart,
            'trace_length': trace_length,
            "time_h": time_h
            }


def derive_heuristic(incidence_matrix, cost_vec, x, t, h):
    x_prime = x.copy()
    x_prime[incidence_matrix.transitions[t]] -= 1
    return max(0, h - cost_vec[incidence_matrix.transitions[t]]), x_prime


def trust_solution(x):
    for v in x:
        if v < -0.001:
            return False
    return True


def vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function):
    ini_vec = incidence_matrix.encode_marking(ini)
    fini_vec = incidence_matrix.encode_marking(fin)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[incidence_matrix.transitions[t]] = cost_function[t]
    return np.array(ini_vec), np.array(fini_vec), np.array(cost_vec)


class IncidenceMatrix(object):

    def __init__(self, net):
        self.__A, self.__B, self.__place_indices, self.__transition_indices = self.__construct_matrix(net)

    def encode_marking(self, marking):
        x = [0 for i in range(len(self.places))]
        for p in marking:
            x[self.places[p]] = marking[p]
        return x

    def __get_a_matrix(self):
        return self.__A

    def __get_b_matrix(self):
        return self.__B

    def __get_transition_indices(self):
        return self.__transition_indices

    def __get_place_indices(self):
        return self.__place_indices

    def __construct_matrix(self, net):
        self.matrix_built = True
        p_index, t_index = {}, {}
        places = sorted([x for x in net.places], key=lambda x: (str(x.name), id(x)))
        transitions = sorted([x for x in net.transitions], key=lambda x: (str(x.name), id(x)))

        for p in places:
            p_index[p] = len(p_index)
        for t in transitions:
            t_index[t] = len(t_index)
        p_index_sort = sorted(p_index.items(), key=lambda kv: kv[0].name, reverse=True)
        t_index_sort = sorted(t_index.items(), key=lambda kv: kv[0].name, reverse=True)
        new_p_index = dict()
        for i in range(len(p_index_sort)):
            new_p_index[p_index_sort[i][0]] = i
        new_t_index = dict()
        for i in range(len(t_index_sort)):
            new_t_index[t_index_sort[i][0]] = i

        a_matrix = np.array([[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))])
        b_matrix = np.array([[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))])
        count = 0
        for p in net.places:
            for a in p.in_arcs:
                a_matrix[new_p_index[p]][new_t_index[a.source]] += 1
            for a in p.out_arcs:
                a_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
                b_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
            # if len(rule_l[count]) == 0 or len(rule_r[count]) == 0:
            #     continue
            # count += 1
        return a_matrix, b_matrix, new_p_index, new_t_index

    a_matrix = property(__get_a_matrix)
    b_matrix = property(__get_b_matrix)
    places = property(__get_place_indices)
    transitions = property(__get_transition_indices)


def construct(net):
    return IncidenceMatrix(net)
