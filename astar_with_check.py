import heapq
import sys
from enum import Enum
import re
import numpy as np
from copy import deepcopy, copy
from pm4py.objects.petri import align_utils as utils
from pm4py.objects.petri.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri.utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from heuristic_past import compute_ini_heuristic, compute_exact_heuristic

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
    return apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function, violate_lst, trace_lst,
                           utils.SKIP, ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                           max_align_time_trace=max_align_time_trace)


def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, violate_lst, trace_lst, skip, ret_tuple_as_trans_desc=False,
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
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1
    trace_sync = [None for i in range(0,len(trace_lst))]
    trace_log = [None for i in range(0,len(trace_lst))]
    t_index = incidence_matrix.transitions
    for t in sync_prod.transitions:
        for i in range(len(trace_lst)):
            if trace_lst[i].name == t.name[0] and t.label[1] == ">>":
                trace_log[i] = t_index[t]
            if trace_lst[i].name == t.name[0] and t.label[1] != ">>":
                trace_sync[i] = t_index[t]
    return __search(sync_prod, initial_marking, final_marking, cost_function, skip, split_dict, incidence_matrix, {},
                    0, 0, visited, queued, traversed, lp_solved, trace_sync, trace_log,
                    ret_tuple_as_trans_desc=ret_tuple_as_trans_desc, use_init=False)


def __search(sync_net, ini, fin, cost_function, skip, split_lst, incidence_matrix, init_dict,
             restart, block_restart, visited, queued, traversed, lp_solved, trace_sync, trace_log,
             ret_tuple_as_trans_desc=False, use_init=False, open_set=None):
    check_set = open_set
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)
    closed = set()
    cost_vec = [x * 1.0 for x in cost_vec]
    cost_vec2 = [x * 1.0 for x in cost_vec]
    t_index = incidence_matrix.transitions
    # t_index2 = {y: x for x, y in t_index.items()}
    set_model_move = []
    for t in t_index:
        if t.label[0] == ">>":
            set_model_move.append(t_index[t])
    p_index = incidence_matrix.places
    if use_init:
        h, x, trustable = init_dict['h'], init_dict['x'], True
    else:
        h, x = compute_exact_heuristic(ini_vec, fin_vec, incidence_matrix.a_matrix, cost_vec2)
    open_set = []
    ini_state = SearchTuple(0 + h, 0, h, ini, None, None, x, True, [])

    # if check_set is not None:
    #     for state in check_set:
    #         state_to_check = get_state(state, ini_state.x, cost_vec, h)
    #         if state_to_check is not None and state_to_check not in open_set:
    #             open_set.append(state_to_check)
    #     print("\ncheck set", ini_state.h, ini_state.x)
    #     for i in open_set:
    #         print(i.trust, i.f, i.g,i.h, i.pre_trans_lst)
        # print("\ncheck set", len(open_set), len(check_set))

    open_set.append(ini_state)
    heapq.heapify(open_set)
    max_events = -1
    old_max = -1
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)
    temp_split = {}
    split_point = None
    dict_g = {ini: 0}
    init_dict = {}
    old_split = None

    #  While not all states visited
    while not len(open_set) == 0:
        # Get the most promising marking
        curr = heapq.heappop(open_set)
        # final marking reached
        if curr.m == fin:
            return utils.__reconstruct_alignment(curr, visited, queued, traversed, restart,
                                                 ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                 lp_solved=lp_solved)

        # heuristic of m is not exact
        if not curr.trust:
            # check if s is not already a splitpoint in K
            if max_events not in split_lst.values() and split_point not in temp_split:
                print("\nopen set", split_lst.values())
                for i in open_set:
                    print(i.trust, i.f, i.g, i.h, i.pre_trans_lst)

                # Add s to the maximum events explained to K
                split_lst.update({split_point: max_events})
                h, x, trustable = compute_ini_heuristic(ini_vec, fin_vec, cost_vec2, incidence_matrix.a_matrix,
                                                        incidence_matrix.b_matrix,split_lst, t_index, p_index,
                                                        trace_sync, trace_log, set_model_move)
                if trustable != 'Optimal':
                    temp_split[split_point] = 1
                    del split_lst[split_point]
                    max_events = old_max
                    split_point = old_split
                    print("Infeasible")
                if np.array_equal(x, ini_state.x):
                    print("Equal solution", split_point, max_events)
                else:
                    init_dict['x'] = x
                    init_dict['h'] = h
                    lp_solved += 1
                    restart += 1
                    return __search(sync_net, ini, fin, cost_function, skip, split_lst, incidence_matrix, init_dict,
                                    restart, block_restart, visited, queued, traversed, lp_solved, trace_sync, trace_log,
                                    ret_tuple_as_trans_desc=False,
                                    use_init=True, open_set=open_set)

            # compute the true heuristic
            h, x = compute_exact_heuristic(incidence_matrix.encode_marking(curr.m),
                                           fin_vec,
                                           incidence_matrix.a_matrix,
                                           cost_vec)
            lp_solved += 1
            if h > curr.h:
                tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True, curr.pre_trans_lst)
                heapq.heappush(open_set, tp)
                heapq.heapify(open_set)
                continue

        closed.add(curr.m)
        new_max_events, last_sync = get_max_events(curr)
        if new_max_events > max_events and last_sync is not None and new_max_events not in split_lst.values():
            old_max = max_events
            max_events = new_max_events
            old_split = split_point
            split_point = last_sync

        visited += 1
        enabled_trans = copy(trans_empty_preset)
        for p in curr.m:
            for t in p.ass_trans:
                if t.sub_marking <= curr.m:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]
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
                h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                trustable = utils.__trust_solution(x)
                new_f = g + h
                pre_trans = deepcopy(curr.pre_trans_lst)
                pre_trans.append(t_index[t])
                tp = SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable, pre_trans)
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
    if marking.t.label[0] != ">>":
        return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())+1, marking.t
    return get_max_events(marking.p)


def get_max_events2(marking):
    if marking.t == None:
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


def get_state(state, ini_vec, cost_vec, h):
    solution_vec = deepcopy(ini_vec)
    curr = state
    for i in range(len(state.pre_trans_lst)):
        solution_vec[state.pre_trans_lst[i]] -= 1

        # when the solution vector encounters -1, means no longer trustable
        if solution_vec[state.pre_trans_lst[i]] < 0:
            solution_vec[state.pre_trans_lst[i]] += 1
            if i > 1:
                for j in range(0, len(state.pre_trans_lst)-i):
                    curr = curr.p
                g = get_g(cost_vec, curr.pre_trans_lst)
                new_h = get_h(h, cost_vec, curr.pre_trans_lst)
                # if h != curr.h:
                #     print("i:", i, "len pre", len(curr.pre_trans_lst), state.pre_trans_lst)
                #     print("h", h, new_h, curr.h, curr.trust)
                f = g + new_h
                check_marking = SearchTuple(f, g, new_h, curr.m, curr.p, curr.t,
                                            solution_vec, True, curr.pre_trans_lst)
                return check_marking
    return None


def get_g(cost_vec, pre_tran_lst):
    g = 0
    for i in pre_tran_lst:
        g += cost_vec[i]
    return g


def get_h(h, cost_vec, pre_tran_lst):
    for i in pre_tran_lst:
        h -= cost_vec[i]
    return h
