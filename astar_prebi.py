import heapq
import sys
from enum import Enum
import re
import numpy as np
from copy import deepcopy, copy
from pm4py.objects.petri import align_utils as utils
# from pm4py.objects.petri.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri.utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util

from construction import construct_cost_aware_backward, construct_cost_aware_forward
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


def apply(trace, petri_net, initial_marking, final_marking, violate_lst_forward, violate_lst_backward, parameters=None):
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
                                violate_lst_forward, violate_lst_backward,
                                parameters)
    return alignment


def apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm,
                    violate_lst_forward, violate_lst_backward,
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

    forward_sync_prod, forward_initial_marking, forward_final_marking, forward_cost_function = \
        construct_cost_aware_forward(trace_net, trace_im, trace_fm, petri_net, initial_marking,
                                     final_marking, utils.SKIP, trace_net_costs, model_cost_function,
                                     revised_sync)

    backward_sync_prod, backward_initial_marking, backward_final_marking, backward_cost_function = \
        construct_cost_aware_backward(trace_net, trace_im, trace_fm, petri_net, initial_marking,
                                      final_marking, utils.SKIP, trace_net_costs, model_cost_function,
                                      revised_sync)
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
    # trace_lst.reverse()
    return apply_sync_prod(forward_sync_prod,
                           forward_initial_marking,
                           forward_final_marking,
                           forward_cost_function,
                           backward_sync_prod,
                           backward_initial_marking,
                           backward_final_marking,
                           backward_cost_function,
                           trace_lst,
                           utils.SKIP, violate_lst_forward, violate_lst_backward,
                           ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                           max_align_time_trace=max_align_time_trace)


def apply_sync_prod(forward_sync_net,
                    forward_initial_marking,
                    forward_final_marking,
                    forward_cost_function,
                    backward_sync_net,
                    backward_initial_marking,
                    backward_final_marking,
                    backward_cost_function, trace_lst, skip,
                    violate_lst_forward, violate_lst_backward,
                    ret_tuple_as_trans_desc=False,
                    max_align_time_trace=sys.maxsize):
    decorate_transitions_prepostset(forward_sync_net)
    decorate_places_preset_trans(forward_sync_net)
    forward_incidence_matrix = construct(forward_sync_net)
    forward_split_dict = violate_lst_forward
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1
    forward_trace_sync = [None for i in range(0, len(trace_lst))]
    forward_trace_log = [None for i in range(0, len(trace_lst))]
    t_index = forward_incidence_matrix.transitions
    for t in forward_sync_net.transitions:
        for i in range(len(trace_lst)):
            if trace_lst[i].name == t.name[0] and t.label[1] == ">>":
                forward_trace_log[i] = t_index[t]
            if trace_lst[i].name == t.name[0] and t.label[1] != ">>":
                forward_trace_sync[i] = t_index[t]
    forward_split = {}
    forward_split1 = list(violate_lst_backward.values())
    for t in backward_sync_net.transitions:
        if t.label[0] == t.label[1] and int(re.search("(\d+)(?!.*\d)", t.name[0]).group()) + 1 in forward_split1:
            forward_split[t] = int(re.search("(\d+)(?!.*\d)", t.name[0]).group()) + 1
    forward_split[None] = -1

    decorate_transitions_prepostset(backward_sync_net)
    decorate_places_preset_trans(backward_sync_net)
    backward_incidence_matrix = construct(backward_sync_net)

    backward_split = {}
    backward_split1 = list(violate_lst_backward.values())
    for t in backward_sync_net.transitions:
        if t.label[0] == t.label[1] and int(re.search("(\d+)(?!.*\d)", t.name[0]).group()) + 1 in backward_split1:
            backward_split[t] = int(re.search("(\d+)(?!.*\d)", t.name[0]).group()) + 1

    backward_split[None] = -1
    trace_lst.reverse()
    backward_trace_sync = [None for i in range(0, len(trace_lst))]
    backward_trace_log = [None for i in range(0, len(trace_lst))]
    t_index = backward_incidence_matrix.transitions
    for t in backward_sync_net.transitions:
        for i in range(len(trace_lst)):
            if trace_lst[i].name == t.name[0] and t.label[1] == ">>":
                backward_trace_log[i] = t_index[t]
            if trace_lst[i].name == t.name[0] and t.label[1] != ">>":
                backward_trace_sync[i] = t_index[t]
    return search(forward_sync_net, forward_initial_marking, forward_final_marking,
                  forward_cost_function, forward_trace_sync, forward_trace_log,
                  {}, forward_incidence_matrix, forward_split,
                  backward_sync_net, backward_initial_marking, backward_final_marking,
                  backward_cost_function, backward_trace_sync, backward_trace_log,
                  {}, backward_incidence_matrix, backward_split,
                  skip, 0, 0, visited, queued, traversed, lp_solved,
                  0,
                  ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                  forward_use_init=False,
                  backward_use_init=False,
                  )


def search(forward_sync_net, forward_ini, forward_fin,
           forward_cost_function, forward_trace_sync, forward_trace_log,
           forward_init_dict, forward_incidence_matrix, forward_split_lst,
           backward_sync_net, backward_ini, backward_fin,
           backward_cost_function, backward_trace_sync, backward_trace_log,
           backward_init_dict, backward_incidence_matrix, backward_split_lst,
           skip, restart, block_restart, visited, queued, traversed, lp_solved,
           direction_count,
           forward_closed={}, backward_closed={},
           ret_tuple_as_trans_desc=False,
           forward_use_init=True,
           backward_use_init=True,
           search_forward=True):
    if not search_forward:
        ini_vec, fin_vec, cost_vec = vectorize_initial_final_cost(
            backward_incidence_matrix, backward_ini, backward_fin,
            backward_cost_function)
        backward_closed = {}
        cost_vec = [x * 1.0 for x in cost_vec]
        cost_vec2 = [x * 1.0 for x in cost_vec]
        t_index = backward_incidence_matrix.transitions
        p_index = backward_incidence_matrix.places
        set_model_move = []
        for t in t_index:
            if t.label[0] == ">>":
                set_model_move.append(t_index[t])
        if backward_use_init:
            h, x, trustable = backward_init_dict['h'], backward_init_dict['x'], True
        elif len(backward_split_lst) > 1:
            h, x, trustable = compute_ini_heuristic(ini_vec, fin_vec, cost_vec2, backward_incidence_matrix.a_matrix,
                                                    backward_incidence_matrix.b_matrix, backward_split_lst, t_index,
                                                    p_index,
                                                    backward_trace_sync, backward_trace_log, set_model_move)
        else:
            h, x = compute_exact_heuristic(ini_vec, fin_vec, backward_incidence_matrix.a_matrix, cost_vec2)
        open_set = []
        order = 0
        ini_state = SearchTuple(0 + h, 0, h, backward_ini, None, None, x, True, [], order)
        open_set.append(ini_state)
        heapq.heapify(open_set)
        max_events = 0
        old_max = 0
        # trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)
        temp_split = {}
        split_point = None
        dict_g = {backward_ini: 0}
        backward_init_dict = {}
        old_split = None

        #  While not all states visited
        while not len(open_set) == 0:
            # Get the most promising marking
            curr = heapq.heappop(open_set)
            # final marking reached
            if curr.m == backward_fin:
                return reconstruct_alignment(curr, visited, queued, traversed, restart, block_restart,
                                             len(backward_trace_log),
                                             ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                             lp_solved=lp_solved)

            # heuristic of m is not exact
            if not curr.trust:

                # check if s is not already a splitpoint in K
                if split_point not in temp_split:
                    # Add s to the maximum events explained to K
                    intersect = backward_closed.keys() & forward_closed.keys()
                    if len(intersect) > 0:
                        for curr in intersect:
                            align1 = reconstruct_alignment(backward_closed[curr], visited, queued, traversed, restart,
                                                           block_restart,
                                                           len(forward_trace_log),
                                                           ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                           lp_solved=lp_solved)
                            align2 = reconstruct_alignment(forward_closed[curr], visited, queued, traversed, restart,
                                                           block_restart,
                                                           len(backward_trace_log),
                                                           ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                           lp_solved=lp_solved)
                            print("forward split", len(forward_split_lst) - 1)
                            print("backward split", len(backward_split_lst) - 1)
                            return reconstruct_alignment2(align2, align1)

                    backward_split_lst.update({split_point: max_events})
                    h, x, trustable = compute_ini_heuristic(ini_vec, fin_vec, cost_vec2,
                                                            backward_incidence_matrix.a_matrix,
                                                            backward_incidence_matrix.b_matrix, backward_split_lst,
                                                            t_index, p_index,
                                                            backward_trace_sync, backward_trace_log, set_model_move)
                    lp_solved += 1
                    if trustable != 'Optimal':
                        temp_split[split_point] = 1
                        del backward_split_lst[split_point]
                        max_events = old_max
                        split_point = old_split
                        print("Infeasible")
                        block_restart += 1
                    # if np.array_equal(x, ini_state.x):
                    #     print("Equal solution", split_point, max_events)
                    #     block_restart += 1
                    else:
                        backward_init_dict['x'] = x
                        backward_init_dict['h'] = h
                        backward_use_init = True
                        direction_count += 1

                        restart += 1
                        direction_count += 1
                        return search(forward_sync_net, forward_ini, forward_fin,
                                      forward_cost_function, forward_trace_sync, forward_trace_log,
                                      forward_init_dict, forward_incidence_matrix, forward_split_lst,
                                      backward_sync_net, backward_ini, backward_fin,
                                      backward_cost_function, backward_trace_sync, backward_trace_log,
                                      backward_init_dict, backward_incidence_matrix, backward_split_lst,
                                      skip, restart, block_restart, visited, queued, traversed, lp_solved,
                                      direction_count,
                                      backward_closed=backward_closed,
                                      backward_use_init=backward_use_init,
                                      forward_use_init=forward_use_init,
                                      search_forward=not search_forward)

                # compute the true heuristic
                h, x = compute_exact_heuristic(backward_incidence_matrix.encode_marking(curr.m),
                                               fin_vec,
                                               backward_incidence_matrix.a_matrix,
                                               cost_vec)
                lp_solved += 1
                if h > curr.h:
                    tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True, curr.pre_trans_lst,
                                     curr.order)
                    heapq.heappush(open_set, tp)
                    heapq.heapify(open_set)
                    continue

            backward_closed[str(curr.m)] = curr
            new_max_events, last_sync = get_max_events(curr)
            if len(backward_trace_log) - new_max_events + 1 > max_events and last_sync is not None \
                    and len(backward_trace_log) - new_max_events + 1 not in backward_split_lst.values():
                old_max = max_events
                max_events = len(backward_trace_log) - new_max_events + 1
                old_split = split_point
                split_point = last_sync

            visited += 1
            enabled_trans = set()
            for p in curr.m:
                for t in p.ass_trans:
                    if t.sub_marking <= curr.m:
                        enabled_trans.add(t)
            trans_to_visit_with_cost = [(t, backward_cost_function[t]) for t in enabled_trans]
            enabled_trans = sorted(sorted(trans_to_visit_with_cost, key=lambda k: k[1]), key=lambda k: k[0].label[0])
            for t, cost in enabled_trans:
                traversed += 1
                new_marking = utils.add_markings(curr.m, t.add_marking)
                if new_marking in backward_closed.keys():
                    continue
                if new_marking not in dict_g:
                    g = curr.g + cost
                    dict_g[new_marking] = g
                    queued += 1
                    h, x = derive_heuristic(backward_incidence_matrix, cost_vec, curr.x, t, curr.h)
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
                                i.h, i.x = derive_heuristic(backward_incidence_matrix, cost_vec, curr.x, t, curr.h)
                                i.trust = trust_solution(i.x)
                                i.f = i.g + i.h
                                i.t = t
                                i.p = curr
                                i.order = curr.order + 1
                                break
            heapq.heapify(open_set)
    else:
        ini_vec, fin_vec, cost_vec = vectorize_initial_final_cost(
            forward_incidence_matrix, forward_ini, forward_fin,
            forward_cost_function)

        forward_closed = {}
        cost_vec = [x * 1.0 for x in cost_vec]
        cost_vec2 = [x * 1.0 for x in cost_vec]
        t_index = forward_incidence_matrix.transitions
        p_index = forward_incidence_matrix.places
        set_model_move = []
        for t in t_index:
            if t.label[0] == ">>":
                set_model_move.append(t_index[t])
        if forward_use_init:
            h, x, trustable = forward_init_dict['h'], forward_init_dict['x'], True
        elif len(forward_split_lst) > 1:
            h, x, trustable = compute_ini_heuristic(ini_vec, fin_vec, cost_vec2, forward_incidence_matrix.a_matrix,
                                                    forward_incidence_matrix.b_matrix, forward_split_lst, t_index,
                                                    p_index,
                                                    forward_trace_sync, forward_trace_log, set_model_move)
        else:
            h, x = compute_exact_heuristic(ini_vec, fin_vec, forward_incidence_matrix.a_matrix, cost_vec2)
        open_set = []
        order = 0
        ini_state = SearchTuple(0 + h, 0, h, forward_ini, None, None, x, True, [], order)
        open_set.append(ini_state)
        heapq.heapify(open_set)
        max_events = 0
        old_max = 0
        # trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)
        temp_split = {}
        split_point = None
        dict_g = {forward_ini: 0}
        forward_init_dict = {}
        old_split = None

        #  While not all states visited
        while not len(open_set) == 0:
            # Get the most promising marking
            curr = heapq.heappop(open_set)
            # final marking reached
            if curr.m == forward_fin:
                return reconstruct_alignment(curr, visited, queued, traversed, restart, block_restart,
                                             len(forward_trace_log),
                                             lp_solved=lp_solved)

            # heuristic of m is not exact
            if not curr.trust:

                # check if s is not already a splitpoint in K
                if max_events not in forward_split_lst.values() and split_point not in temp_split:
                    # Add s to the maximum events explained to K
                    intersect = backward_closed.keys() & forward_closed.keys()
                    if len(intersect) > 0:
                        for curr in intersect:
                            align1 = reconstruct_alignment(backward_closed[curr], visited, queued, traversed, restart,
                                                           block_restart,
                                                           len(forward_trace_log),
                                                           ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                           lp_solved=lp_solved)
                            align2 = reconstruct_alignment(forward_closed[curr], visited, queued, traversed, restart,
                                                           block_restart,
                                                           len(backward_trace_log),
                                                           ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                           lp_solved=lp_solved)
                            print("forward split", len(forward_split_lst) - 1)
                            print("backward split", len(backward_split_lst) - 1)
                            return reconstruct_alignment2(align2, align1)

                    forward_split_lst.update({split_point: max_events})
                    h, x, trustable = compute_ini_heuristic(ini_vec, fin_vec, cost_vec2,
                                                            forward_incidence_matrix.a_matrix,
                                                            forward_incidence_matrix.b_matrix, forward_split_lst,
                                                            t_index, p_index,
                                                            forward_trace_sync, forward_trace_log, set_model_move)
                    lp_solved += 1
                    if trustable != 'Optimal':
                        temp_split[split_point] = 1
                        del forward_split_lst[split_point]
                        max_events = old_max
                        split_point = old_split
                        print("Infeasible")
                        block_restart += 1
                    # if np.array_equal(x, ini_state.x):
                    #     print("Equal solution", split_point, max_events)
                    #     block_restart += 1
                    else:
                        forward_init_dict['x'] = x
                        forward_init_dict['h'] = h
                        forward_use_init = True
                        direction_count += 1
                        restart += 1
                        return search(forward_sync_net, forward_ini, forward_fin,
                                      forward_cost_function, forward_trace_sync, forward_trace_log,
                                      forward_init_dict, forward_incidence_matrix, forward_split_lst,
                                      backward_sync_net, backward_ini, backward_fin,
                                      backward_cost_function, backward_trace_sync, backward_trace_log,
                                      backward_init_dict, backward_incidence_matrix, backward_split_lst,
                                      skip, restart, block_restart, visited, queued, traversed, lp_solved,
                                      direction_count,
                                      forward_closed=forward_closed,
                                      forward_use_init=forward_use_init,
                                      backward_use_init=backward_use_init,
                                      search_forward=not search_forward)

                # compute the true heuristic
                h, x = compute_exact_heuristic(forward_incidence_matrix.encode_marking(curr.m),
                                               fin_vec,
                                               forward_incidence_matrix.a_matrix,
                                               cost_vec)
                lp_solved += 1
                if h > curr.h:
                    tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True, curr.pre_trans_lst,
                                     curr.order)
                    heapq.heappush(open_set, tp)
                    heapq.heapify(open_set)
                    continue

            forward_closed[str(curr.m)] = curr
            new_max_events, last_sync = get_max_events(curr)
            if new_max_events > max_events and last_sync is not None \
                    and new_max_events not in forward_split_lst.values():
                old_max = max_events
                max_events = new_max_events
                old_split = split_point
                split_point = last_sync

            visited += 1
            enabled_trans = set()
            for p in curr.m:
                for t in p.ass_trans:
                    if t.sub_marking <= curr.m:
                        enabled_trans.add(t)
            trans_to_visit_with_cost = [(t, forward_cost_function[t]) for t in enabled_trans]
            enabled_trans = sorted(sorted(trans_to_visit_with_cost, key=lambda k: k[1]), key=lambda k: k[0].label[0])
            for t, cost in enabled_trans:
                traversed += 1
                new_marking = utils.add_markings(curr.m, t.add_marking)
                if new_marking in forward_closed.keys():
                    continue
                if new_marking not in dict_g:
                    g = curr.g + cost
                    dict_g[new_marking] = g
                    queued += 1
                    h, x = derive_heuristic(forward_incidence_matrix, cost_vec, curr.x, t, curr.h)
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
                                i.h, i.x = derive_heuristic(forward_incidence_matrix, cost_vec, curr.x, t, curr.h)
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


def reconstruct_alignment(state, visited, queued, traversed, restart, block_restart, trace_length,
                          ret_tuple_as_trans_desc=False, lp_solved=0):
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
                # print("state", parent.p.m)
                alignment = [parent.t.label] + alignment
                parent = parent.p
    return {'alignment': alignment,
            'cost': state.g,
            'visited_states': visited,
            'queued_states': queued,
            'traversed_arcs': traversed,
            'lp_solved': lp_solved,
            'restart': restart,
            'block_restart': block_restart,
            'trace_length': trace_length
            }


def reconstruct_alignment2(rec1, rec2):
    return {'alignment': rec1['alignment'] + rec2['alignment'],
            'cost': rec1['cost'] + rec2['cost'],
            'visited_states': rec1['visited_states'],
            'queued_states': rec1['queued_states'],
            'traversed_arcs': rec1['traversed_arcs'],
            'lp_solved': rec1['lp_solved'],
            'restart': rec1['restart'],
            'block_restart': rec1['block_restart'],
            'trace_length': rec1['trace_length']
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
    return ini_vec, fini_vec, cost_vec


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

    # def __get_rule_l(self):
    #     return self.__rule_l
    #
    # def __get_rule_r(self):
    #     return self.__rule_r

    def __get_transition_indices(self):
        return self.__transition_indices

    def __get_place_indices(self):
        return self.__place_indices

    def __construct_matrix(self, net):
        self.matrix_built = True
        p_index, t_index = {}, {}
        places = sorted([x for x in net.places], key=lambda x: (str(x.name), id(x)))
        transitions = sorted([x for x in net.transitions], key=lambda x: (str(x.name), id(x)))
        rule_l = {}
        rule_r = {}

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

        a_matrix = [[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))]
        b_matrix = [[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))]
        count = 0
        for p in net.places:
            rule_l[count] = set()
            rule_r[count] = set()
            for a in p.in_arcs:
                a_matrix[new_p_index[p]][new_t_index[a.source]] += 1
                rule_l[count].add(a.source.label)
            for a in p.out_arcs:
                a_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
                b_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
                rule_r[count].add(a.target.label)
            # if len(rule_l[count]) == 0 or len(rule_r[count]) == 0:
            #     continue
            # count += 1
        return a_matrix, b_matrix, new_p_index, new_t_index

    a_matrix = property(__get_a_matrix)
    b_matrix = property(__get_b_matrix)
    places = property(__get_place_indices)
    transitions = property(__get_transition_indices)
    # rule_l = property(__get_rule_l)
    # rule_r = property(__get_rule_r)


def construct(net):
    return IncidenceMatrix(net)
