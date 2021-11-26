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
from heuristic import get_ini_heuristic, get_exact_heuristic


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
                model_cost_function[t] = 0
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
    # print(trace_lst)
    return apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function, trace_lst)


def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, trace_lst):
    # Get the syncronous product net
    decorate_transitions_prepostset(sync_prod)
    decorate_places_preset_trans(sync_prod)
    sync_prod_net = construct(sync_prod)

    start_time = timeit.default_timer()
    res = search(sync_prod_net, initial_marking, final_marking, cost_function, trace_lst)

    # Get the total running time
    res['time_sum'] = round(timeit.default_timer() - start_time, 6)

    # Get the total search time
    res['time_diff'] = round(res['time_sum'] - res['time_h'], 6)
    print(res)
    return res


def prove_validity(state, ini_vec):
    solution_vec = deepcopy(ini_vec)
    flag = True
    for j in state.pre_trans_lst:
        solution_vec[j] -= 1
        # when the solution vector encounters -1, means no longer trustable
        if solution_vec[j] < 0:
            # print(j, state.pre_trans_lst)
            # trust = False
            flag = False
            break
    return flag


def path_propagate(subsequent_states, open_set):
    # iterate states in open set and add new paths to them
    for state in open_set:
        if state.m in subsequent_states:
            path_propagate(state, open_set)
    return open_set


def search(sync_prod_net, ini, fin, cost_function, trace_lst):
    ini_vec, fin_vec, cost_vec = vectorize_initial_final_cost(sync_prod_net, ini, fin, cost_function)
    time_h, queued, visited, traversed, lp_solved, restart = 0, 0, 0, 0, 0, 0

    p_index = sync_prod_net.places
    t_index = sync_prod_net.transitions
    incidence_matrix = sync_prod_net.a_matrix
    consumption_matrix = sync_prod_net.b_matrix
    trace_sync = [[] for i in range(len(trace_lst))]
    trace_log = [None for i in range(len(trace_lst))]

    # Get the list of index for trace move and synchronous move, will be used for linear programming
    for t in t_index:
        for i in range(len(trace_lst)):
            if trace_lst[i].name == t.name[0] and t.label[1] == ">>":
                trace_log[i] = t_index[t]
            if trace_lst[i].name == t.name[0] and t.label[1] != ">>":
                trace_sync[i].append(t_index[t])

    start_time = timeit.default_timer()
    marking_diff = fin_vec - ini_vec
    h, x = get_exact_heuristic(marking_diff, incidence_matrix, cost_vec)
    solution_vec = [x]
    time_h += timeit.default_timer() - start_time
    lp_solved += 1
    ini_state = SearchTuple(0 + h, 0, h, ini, None, None, solution_vec, True, [[]])
    open_set = [ini_state]
    cache_set = []
    heapq.heapify(open_set)
    already_visited = {ini: 0}
    max_events = -1
    split_lst = []
    closed = {}

    while True:
        # While not all states visited
        while open_set:
            # get the most promising marking
            curr = heapq.heappop(open_set)

            # final marking reached
            if curr.m == fin:
                # print(curr.pre_trans_lst)
                return reconstruct_alignment(curr, visited, queued, traversed, lp_solved, restart, time_h)

            # Heuristic of m is not exact, add to cache set and continue
            if not curr.trust:
                cache_set.append(curr)
                continue

            # if the solution vector is valid, we can expand
            visited += 1
            # Add current marking to closed set
            closed[curr.m] = curr

            # Update max events explained
            new_max_events = get_max_events(curr)
            if new_max_events > max_events:
                max_events = new_max_events

            enabled_trans = set()
            for p in curr.m:
                for t in p.ass_trans:
                    if t.sub_marking <= curr.m:
                        enabled_trans.add(t)

            trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans]
            # if len(split_lst) == 9:
            # print("\ncurr", curr.m, curr.pre_trans_lst, "\n", curr.x)

            for t, cost in trans_to_visit_with_cost:
                traversed += 1
                new_marking = utils.add_markings(curr.m, t.add_marking)
                traversed += 1
                g = curr.g + cost
                h, x = derive_heuristic(curr.x, curr.h, t_index[t], cost)
                trustable = trust_solution(x)

                # if the state is in closed, we need to check whether new possible paths leading to it
                if new_marking in closed:
                    # add possible new path to closed set
                    flag = False
                    pre_trans = deepcopy(curr.pre_trans_lst)
                    for j in pre_trans:
                        j.append(t_index[t])

                    # use mark to indicate which one in pre_trans to append
                    mark = [1 for i in range(len(pre_trans))]
                    count = 0
                    for j in pre_trans:
                        for k in closed[new_marking].pre_trans_lst:
                            if len(j) == len(k) and set(j) == set(k):
                                mark[count] = 0
                                continue
                        count += 1
                    for i in range(len(mark)):
                        if mark[i] == 1:
                            flag = True
                            closed[new_marking].pre_trans_lst.append(pre_trans[i])
                    # print("in closed", t_index[t],  trustable, new_marking, trustable, closed[new_marking].pre_trans_lst)

                    # no new paths, continue
                    if flag:
                        update_tp = closed[new_marking]
                        update_tp.trust = trustable
                        update_tp.h = h
                        update_tp.x = deepcopy(x)
                        update_tp.g = min(update_tp.g, g)
                        update_tp.t = t
                        update_tp.f = update_tp.g + update_tp.h
                        heapq.heappush(open_set, update_tp)
                        del closed[new_marking]
                        already_visited[new_marking] = min(update_tp.g, g)

                # new marking is fresh or find shorter path, add it to open set
                elif new_marking not in already_visited:
                    already_visited[new_marking] = g
                    pre_trans = deepcopy(curr.pre_trans_lst)
                    for j in pre_trans:
                        j.append(t_index[t])
                    tp = SearchTuple(g + h, g, h, new_marking, curr, t, deepcopy(x), trustable, pre_trans)
                    # if len(split_lst) == 9:
                    # print("not visited", t_index[t], trustable, new_marking, trustable, pre_trans, "\n", deepcopy(x))
                    queued += 1
                    heapq.heappush(open_set, tp)
                    continue

                # new marking has shorter path
                elif g <= already_visited[new_marking]:
                    already_visited[new_marking] = g

                    for i in open_set:
                        if i.m == new_marking:
                            pre_trans = deepcopy(curr.pre_trans_lst)
                            for j in pre_trans:
                                j.append(t_index[t])

                            mark = [1 for i in range(len(pre_trans))]
                            count = 0
                            for j in pre_trans:
                                for k in i.pre_trans_lst:
                                    if len(j) == len(k) and set(j) == set(k):
                                        mark[count] = 0
                                        continue
                                count += 1
                            for ele in range(len(mark)):
                                if mark[ele] == 1:
                                    i.pre_trans_lst.append(pre_trans[ele])
                            # if len(split_lst) == 9:
                            # print("smaller or equal g", t_index[t], trustable, new_marking, i.trust, trustable, i.pre_trans_lst)
                            i.g = g
                            # 这里是个问题哦！！
                            if trustable and not i.trust:
                                i.h, i.x = h, deepcopy(x)
                                i.trust = trustable or i.trust
                            i.f = i.g + i.h
                            i.t = t
                            i.p = curr
                            break

                # new marking has longer path, but the heuristic change from invalid to valid
                else:
                    # print("longer", new_marking, trustable)
                    if trustable:
                        for i in open_set:
                            if i.m == new_marking:
                                if not i.trust:
                                    i.h, i.x = h, x
                                    i.f = i.g + i.h
                                    i.trust = True

        # check if s is not already a split point in K
        if max_events + 1 not in split_lst and max_events < len(trace_lst) - 1:
            split_lst.append(max_events + 1)
            start_time = timeit.default_timer()
            splits = sorted(split_lst)
            h, x = get_ini_heuristic(ini_vec, fin_vec, cost_vec, splits,
                                     incidence_matrix, consumption_matrix,
                                     t_index, p_index, trace_sync, trace_log)
            print(splits, h, len(cache_set), len(closed))

            time_h += timeit.default_timer() - start_time
            lp_solved += 1
            restart += 1
            open_set = []

            # print("closed")
            # if len(split_lst) == 9:
            #     for k,v in closed.items():
            #         print(v.m, v.pre_trans_lst)

            # use flag to indicate whether valid states exist
            flag = True

            for i in cache_set:
                new_i = get_state(i, x, cost_vec, h)
                open_set.append(new_i)
                if new_i.trust:
                    # print("become true")
                    flag = False

            cache_set = []
            max_events = -1

            if flag:
                # print("restart")
                closed = {}
                solution_vec = [x]
                ini_state = SearchTuple(0 + h, 0, h, ini, None, None, solution_vec, True, [[]])
                open_set = [ini_state]
                cache_set = []
                heapq.heapify(open_set)
                already_visited = {ini: 0}
            heapq.heapify(open_set)


def get_state(state, ini_vec, cost_vec, h):
    trust = False
    bool_lst = [1 for i in range(len(state.pre_trans_lst))]
    count = 0
    state.x = []
    for j in state.pre_trans_lst:
        solution_vec = deepcopy(ini_vec)
        for i in range(len(j)):
            solution_vec[j[i]] -= 1
            # when the solution vector encounters -1, means no longer trustable
            if solution_vec[j[i]] < 0:
                # trust = False
                bool_lst[count] = 0
                continue
        if bool_lst[count] == 1:
            state.x.append(solution_vec)
            trust = True
        count += 1
    new_h = get_h(h, cost_vec, state.pre_trans_lst)
    state.f = state.g + h
    state.h = new_h
    state.trust = trust
    return state


def get_g(cost_vec, pre_tran_lst):
    g = 0
    for i in pre_tran_lst:
        g += cost_vec[i]
    return g


def get_h(h, cost_vec, pre_tran_lst):
    h_lst = [0]
    for j in pre_tran_lst:
        for i in j:
            h -= cost_vec[i]
        h_lst.append(h)
    return max(h_lst)


class SearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust, pre_trans_lst):
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
        if self.f < other.f:
            return True
        elif other.f < self.f:
            return False
        if self.trust != other.trust:
            return self.trust
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
        return -1
    if marking.t.label[0] != ">>":
        return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())
    return get_max_events(marking.p)


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


def reconstruct_alignment(state, visited, queued, traversed, lp_solved, restart, time_h, ret_tuple_as_trans_desc=False):
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
    return {'cost': state.g,
            'visited_states': visited,
            'queued_states': queued,
            'traversed_arcs': traversed,
            'lp_solved': lp_solved,
            'restart': restart,
            "time_h": round(time_h, 6),
            'alignment': alignment
            }


def derive_heuristic(x, h, t_index, cost):
    y = deepcopy(x)
    for i in y:
        i[t_index] -= 1
    return max(0, h - cost), y


def trust_solution(x):
    trust = [1 for i in range(len(x))]
    count = 0
    for j in x:
        for k in j:
            if k < 0:
                trust[count] = 0
                break
        count += 1
    if 1 in trust:
        return True
    else:
        return False


def vectorize_initial_final_cost(sync_prod_net, ini, fin, cost_function):
    ini_vec = sync_prod_net.encode_marking(ini)
    fini_vec = sync_prod_net.encode_marking(fin)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[sync_prod_net.transitions[t]] = cost_function[t]
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
        for p in net.places:
            for a in p.in_arcs:
                a_matrix[new_p_index[p]][new_t_index[a.source]] += 1
            for a in p.out_arcs:
                a_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
                b_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
        return a_matrix, b_matrix, new_p_index, new_t_index

    a_matrix = property(__get_a_matrix)
    b_matrix = property(__get_b_matrix)
    places = property(__get_place_indices)
    transitions = property(__get_transition_indices)


def construct(net):
    return IncidenceMatrix(net)
