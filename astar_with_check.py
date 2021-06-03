import heapq
import sys
import time
from copy import copy
from enum import Enum
import re
import numpy as np

from pm4py import util as pm4pyutil
from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri import align_utils as utils
from pm4py.objects.petri.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri.utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.lp import solver as lp_solver
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from astar_implementation.heuristic import compute_ini_heuristic
# from astar_implementation.incidence_matrix import construct as inc_mat_construct

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


def apply(trace, petri_net, initial_marking, final_marking, violate_lst, parameters=None):
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

    if trace_net_constr_function is not None:
        # keep the possibility to pass TRACE_NET_CONSTR_FUNCTION in this old version
        trace_net, trace_im, trace_fm = trace_net_constr_function(trace, activity_key=activity_key)
    else:
        trace_net, trace_im, trace_fm, parameters[
            Parameters.PARAM_TRACE_NET_COSTS] = trace_net_cost_aware_constr_function(trace,
                                                                                     trace_cost_function,
                                                                                     activity_key=activity_key)
    alignment = apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, violate_lst, parameters)
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


def apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, violate_lst, parameters=None):
    if parameters is None:
        parameters = {}

    ret_tuple_as_trans_desc = exec_utils.get_param_value(Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE,
                                                         parameters, False)

    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    sync_cost_function = exec_utils.get_param_value(Parameters.PARAM_SYNC_COST_FUNCTION, parameters, None)
    trace_net_costs = exec_utils.get_param_value(Parameters.PARAM_TRACE_NET_COSTS, parameters, None)

    if trace_cost_function is None or model_cost_function is None or sync_cost_function is None:
        sync_prod, sync_initial_marking, sync_final_marking = construct(trace_net, trace_im,
                                                                        trace_fm, petri_net,
                                                                        initial_marking,
                                                                        final_marking,
                                                                        utils.SKIP)
        cost_function = utils.construct_standard_cost_function(sync_prod, utils.SKIP)
    else:
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

    return apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function,violate_lst,
                           utils.SKIP, ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                           max_align_time_trace=max_align_time_trace)


def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, violate_lst, skip, ret_tuple_as_trans_desc=False,
                    max_align_time_trace=sys.maxsize):
    decorate_transitions_prepostset(sync_prod)
    decorate_places_preset_trans(sync_prod)

    incidence_matrix = inc_mat_construct(sync_prod)
    split_dict = {}

    violate = list(violate_lst.values())
    for t in sync_prod.transitions:
        if t.label[0] == t.label[1] and int(re.search("(\d+)(?!.*\d)", t.name[0]).group())+1 in violate:
            split_dict[t] = int(re.search("(\d+)(?!.*\d)", t.name[0]).group())+1
    split_dict[None] = -1
    cache_set = set()
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1
    # print("split dict", split_dict)
    return __search(sync_prod, initial_marking, final_marking, cost_function, skip, split_dict, incidence_matrix, {},
                    0, 0, visited, queued, traversed, lp_solved,
                    ret_tuple_as_trans_desc=ret_tuple_as_trans_desc, use_init=False)


def __search(sync_net, ini, fin, cost_function, skip, split_lst, incidence_matrix, init_dict,
             restart, block_restart, visited, queued, traversed, lp_solved,
             ret_tuple_as_trans_desc=False, use_init=False):
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)
    closed = set()
    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    g_matrix = -np.eye(len(sync_net.transitions))
    h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()
    cost_vec = [x * 1.0 for x in cost_vec]
    cost_vec2 = [x * 1.0 for x in cost_vec]

    use_cvxopt = False
    if lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN or lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP:
        use_cvxopt = True

    if use_cvxopt:
        # not available in the latest version of PM4Py
        from cvxopt import matrix

        a_matrix = matrix(a_matrix)
        g_matrix = matrix(g_matrix)
        h_cvx = matrix(h_cvx)
        cost_vec = matrix(cost_vec)

    t_index = incidence_matrix.transitions
    if len(split_lst) == 1:
        print("transition index:", t_index)
    p_index = incidence_matrix.places
    order = 0
    if use_init == True:
        h, x, trustable = init_dict['h'], init_dict['x'], True
    else:
        h, x, trustable = compute_ini_heuristic(ini_vec, fin_vec, cost_vec2, incidence_matrix.a_matrix, incidence_matrix.b_matrix, split_lst, t_index, p_index)

    open_set = []
    ini_state = SearchTuple(0 + h, 0, h, ini, None, None, x, True)

    open_set.append(ini_state)
    heapq.heapify(open_set)

    max_events = -1
    old_max = 0
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)
    temp_split = {}
    split_point = None
    dict_g = {}
    dict_g[ini] = 0
    order += 1
    init_dict = {}
    replay = True

    #  While not all states visited
    while not len(open_set) == 0:

        # Get the most promising marking
        curr = heapq.heappop(open_set)

        # final marking reached
        if curr.m == fin:
            # print(len(split_lst))
            return utils.__reconstruct_alignment(curr, visited, queued, traversed, restart,
                                                 ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                 lp_solved=lp_solved)

        # heuristic of m is not exact
        if not curr.trust:
            # print(split_point, get_pre_trans(curr,[]))
            # check if s is not already a splitpoint in K
            if split_point not in split_lst and split_point not in temp_split and replay:

                # Add s to the maximum events explained to K
                split_lst.update({split_point: max_events})

                h, x, trustable = compute_ini_heuristic(ini_vec, fin_vec, cost_vec2, incidence_matrix.a_matrix,
                                                        incidence_matrix.b_matrix,split_lst, t_index, p_index)

                if trustable != 'Optimal':
                    print("remove 1:", split_point, split_lst, max_events)
                    temp_split[split_point] = 1
                    del split_lst[split_point]
                    max_events = -1
                    split_point = None
                    # replay = False
                # else:
                #     if np.array_equal(x, ini_state.x):
                #         print("remove 2:", split_point, max_events)
                #         temp_split[split_point] = 1
                #         del split_lst[split_point]
                #         max_events = -1
                #         split_point = None
                #         block_restart += 1
                else:
                    init_dict['x'] = x
                    init_dict['h'] = h
                    restart += 1
                    for i in open_set:
                        i.pre_trans_lst = get_pre_trans(i, [])
                        i.h, i.x = derive_heuristic_for_op(incidence_matrix, cost_vec, x, i.pre_trans_lst, h)
                        i.trust = utils.__trust_solution(i.x)
                        i.f = i.g + i.h
                    print('restart', split_point, max_events)
                    return __search(sync_net, ini, fin, cost_function, skip, split_lst, incidence_matrix, init_dict,
                                    restart, block_restart, visited, queued, traversed, lp_solved, open_set,
                                    ret_tuple_as_trans_desc=False,
                                    use_init=True)

            # compute the true heuristic
            h, x = utils.__compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                               incidence_matrix, curr.m,
                                                               fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                               use_cvxopt=use_cvxopt)

            lp_solved += 1

            if h > curr.h:
                tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
                heapq.heappush(open_set, tp)
                heapq.heapify(open_set)
                continue

        closed.add(curr.m)
        if len(split_lst) == 7 and curr.t is not None:
            if curr.t.label[0] == curr.t.label[1] == '01_HOOFD_200':
                print(curr.m, get_pre_events(curr,[]))
        new_max_events, last_sync = get_max_events(curr)

        if new_max_events > max_events and last_sync is not None:
            old_max = max_events
            max_events = new_max_events
            old_split_point = split_point
            split_point = last_sync


        visited += 1

        enabled_trans = copy(trans_empty_preset)
        for p in curr.m:
            for t in p.ass_trans:
                if t.sub_marking <= curr.m:
                    enabled_trans.add(t)

        # trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans]
        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]
        enabled_trans = sorted(sorted(trans_to_visit_with_cost, key=lambda k: k[1]), key=lambda k: k[0].label[0])

        for t, cost in enabled_trans:

            traversed += 1
            new_marking = utils.add_markings(curr.m, t.add_marking)

            if new_marking not in dict_g:
                g = curr.g + cost
                dict_g[new_marking] = g
                queued += 1
                h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                trustable = utils.__trust_solution(x)
                new_f = g + h
                tp = SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
                heapq.heappush(open_set, tp)
                # if t.label[0] == '04_BPT_005' and t.label[1] == '04_BPT_005':
                #     print(len(split_lst),curr.t, tp.t, tp.f, tp.g, tp.trust, get_pre_events(tp, []))
                # if len(split_lst) == 7 and t.label[0] == '04_BPT_005':
                #     print(curr.t, tp.t, tp.f, tp.g, tp.trust, get_pre_events(tp, []))

            else:
                if curr.g + cost < dict_g[new_marking]:
                    dict_g[new_marking] = curr.g + cost
                    if new_marking in closed:
                        closed.remove(new_marking)
                        g = curr.g + cost
                        queued += 1
                        h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                        trustable = utils.__trust_solution(x)
                        new_f = g + h
                        tp = SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
                        heapq.heappush(open_set, tp)
                        # if len(split_lst) == 7 and t.label[0] == '04_BPT_005':
                        # #     print(tp.t, tp.trust, get_pre_events(tp, []))
                    else:
                        for i in open_set:
                            if i.m == new_marking:
                                i.g = curr.g + cost
                                queued += 1
                                i.h, i.x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                                i.trust = utils.__trust_solution(i.x)
                                i.f = i.g + i.h
                                i.t = t
                                i.p = curr
                                break

        heapq.heapify(open_set)


class SearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust, pre_trans_lst=None):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        self.pre_trans_lst = pre_trans_lst


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
        # if self.f < other.f:
        #     return True
        # elif other.f < self.f:
        #     return False
        # elif self.trust and not other.trust:
        #     return True
        # else:
        #     return self.h < other.h


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
    if marking.t == None:
        return 0, None
    if marking.t.label[0] == marking.t.label[1]:
        return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())+1, marking.t
    return get_max_events(marking.p)

def get_max_events2(marking):
    if marking.t == None:
        return 0
    if marking.t.label[0] == marking.t.label[1]:
        # return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group()), marking.t
        return 1+ get_max_events2(marking.p)


def get_path_length(marking):
    if marking.p == None:
        return 0
    else:
        return 1 + get_path_length(marking.p)


def get_pre_events(marking, lst):
    if marking.t == None:
        return lst
    lst.insert(0, marking.t.label)
    return get_pre_events(marking.p, lst)


def get_pre_trans(marking, lst):
    if marking.t == None:
        return lst
    lst.insert(0, marking.t)
    return get_pre_trans(marking.p, lst)


def derive_heuristic_for_op(incidence_matrix, cost_vec, x, pre_t, h):
    x_prime = x.copy()
    for t in pre_t:
        x_prime[incidence_matrix.transitions[t]] -= 1
        if t.label[0] != t.label[1]:
            h -= 1
    return max(0, h), x_prime

