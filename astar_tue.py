import construction
import minheap
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
from heuristic import get_ini_heuristic, get_exact_heuristic_new, get_exact_heuristic


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


class Inc_astar:

    def __init__(self, trace, petri_net, initial_marking, final_marking):
        self.trace = trace
        self.petri_net = petri_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.visited = 0
        self.queued = 0
        self.traversed = 0
        self.lp_solved = 0
        self.restart = 0
        self.max_rank = -1
        self.time_h = 0
        self.open_set = MinHeap()
        self.time_heap = 0
        self.heap_insert = 0
        self.heap_retrieval = 0
        self.heap_delete = 0
        self.heap_pop = 0
        self.time_sort = 0
        self.order = 0
        self.split_lst = []
        self.heap_count = 0
        self.heap_longtime = 0

    def apply(self, trace, petri_net, initial_marking, final_marking, parameters=None):
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
        trace_net_cost_aware_constr_function = exec_utils.get_param_value(
            Parameters.TRACE_NET_COST_AWARE_CONSTR_FUNCTION,
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
                    model_cost_function[t] = 1
                    sync_cost_function[t] = 0
                else:
                    model_cost_function[t] = 0
            parameters[Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
            parameters[Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function
        trace_net, trace_im, trace_fm, parameters[
            Parameters.PARAM_TRACE_NET_COSTS] = trace_net_cost_aware_constr_function(trace,
                                                                                     trace_cost_function,
                                                                                     activity_key=activity_key)
        alignment = self.apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm,
                                         parameters)
        return alignment

    def apply_trace_net(self, petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm,
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

        sync_prod, sync_initial_marking, sync_final_marking, cost_function = construction.construct_cost_aware_forward(
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
        # print("trace lst:",trace_lst)
        return self.apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function, trace_lst)

    def apply_sync_prod(self, sync_prod, initial_marking, final_marking, cost_function, trace_lst):
        decorate_transitions_prepostset(sync_prod)
        decorate_places_preset_trans(sync_prod)

        incidence_matrix = construct(sync_prod)
        trace_sync = [[] for i in range(0, len(trace_lst))]
        trace_log = [None for i in range(0, len(trace_lst))]
        t_index = incidence_matrix.transitions
        for t in sync_prod.transitions:
            for i in range(len(trace_lst)):
                if trace_lst[i].name == t.name[0] and t.label[1] == ">>":
                    trace_log[i] = t_index[t]
                if trace_lst[i].name == t.name[0] and t.label[1] != ">>":
                    trace_sync[i].append(t_index[t])
        start_time = timeit.default_timer()
        res = self.search(initial_marking, final_marking, cost_function, incidence_matrix,
                          trace_sync, trace_log)
        res['time_sum'] = timeit.default_timer() - start_time
        res['time_diff'] = res['time_sum'] - res['time_h']
        return res

    def search(self, ini, fin, cost_function, incidence_matrix, trace_sync, trace_log):
        ini_vec, fin_vec, cost_vec = self.vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)
        closed = set()
        p_index = incidence_matrix.places
        inc_matrix = incidence_matrix.a_matrix
        cons_matrix = incidence_matrix.b_matrix
        start_time = timeit.default_timer()
        h, x = get_exact_heuristic(fin_vec - ini_vec, inc_matrix, cost_vec)
        self.time_h += timeit.default_timer() - start_time
        self.lp_solved += 1

        # add initial marking to open set
        ini_state = Marking(h, 0, h, ini, None, None, deepcopy(x), True, self.order)
        start_time = timeit.default_timer()
        self.open_set.heap_insert(ini_state)
        if timeit.default_timer() - start_time > 0.005:
            # print("p1")
            self.heap_longtime += 1
        self.time_heap += timeit.default_timer() - start_time
        self.heap_insert += 1
        self.heap_count += 1
        # while not all states visited
        while self.open_set:
            # get the most promising marking
            self.visited += 1
            start_time = timeit.default_timer()
            new_curr = self.open_set.heap_pop()
            # print("\n", self.visited)
            if timeit.default_timer() - start_time > 0.005:
                # print("p2")
                self.heap_longtime += 1

            self.time_heap += timeit.default_timer() - start_time
            self.heap_pop += 1
            self.heap_count += 1
            marking_diff = fin_vec - incidence_matrix.encode_marking(new_curr.m)
            curr, flag = \
                self.close_or_update_marking(new_curr, ini_state.m, fin, cost_vec,
                                             marking_diff, inc_matrix, len(trace_log))
            if flag == "CLOSEDSUCCESSFUL":
                closed.add(curr.m)
                # only after adding marking m to closed set, can we expand from m
                self.expand_marking(cost_function, curr, incidence_matrix, cost_vec, closed)
            elif flag == "REQUEUED":
                start_time = timeit.default_timer()
                self.open_set.heap_insert(curr)
                if timeit.default_timer() - start_time > 0.005:
                    # print("p3")
                    self.heap_longtime += 1

                self.time_heap += timeit.default_timer() - start_time
                self.heap_count += 1
                self.heap_insert += 1
            elif flag == "RESTARTNEEDED":
                self.split_lst = sorted(self.split_lst)
                start_time = timeit.default_timer()
                h, x = get_ini_heuristic(ini_vec, fin_vec, cost_vec, self.split_lst, inc_matrix, cons_matrix,
                                         incidence_matrix.transitions, p_index,
                                         trace_sync, trace_log)
                print("split_lst", self.split_lst, h)
                self.time_h += timeit.default_timer() - start_time
                self.lp_solved += 1
                self.restart += 1
                # restart by reset open set and closed set
                closed = set()
                self.order = 0
                ini_state = Marking(h, 0, h, ini, None, None, deepcopy(x), True, self.order)
                start_time = timeit.default_timer()
                self.open_set.heap_clear()
                self.open_set.heap_insert(ini_state)
                if timeit.default_timer() - start_time > 0.005:
                    print("p4")
                    self.heap_longtime += 1

                self.time_heap += timeit.default_timer() - start_time
                self.heap_count += 1
                self.heap_insert += 1
                self.max_rank = -1
            elif flag == "FINALMARKINGFOUND":
                print("long time", self.heap_longtime)
                return self.reconstruct_alignment(curr, len(trace_log))
            elif flag == "CLOSEDINFEASIBLE":
                closed[curr.m] = curr

    def close_or_update_marking(self, marking, ini, fin, cost_vec, marking_diff,
                                incidence_matrix, len_trace):
        if marking.m == fin and marking.trust:
            return marking, "FINALMARKINGFOUND"
        # if the heuristic is not exact
        if not marking.trust:
            # compute the exact heuristics
            self.split_lst = sorted(self.split_lst)
            start_time = timeit.default_timer()
            h, x, trustable, self.split_lst, self.max_rank = \
                get_exact_heuristic_new(marking, self.split_lst, marking_diff, ini, incidence_matrix,
                                        cost_vec, self.max_rank, len_trace)
            self.time_h += timeit.default_timer() - start_time
            # need to restart
            if h == -1:
                return marking, "RESTARTNEEDED"
            # heuristic is not computable, from which final marking is unreachable
            elif trustable == "Infeasible":
                self.lp_solved += 1
                return marking, "CLOSEDINFEASIBLE"
            # if the heuristic is higher push the head of the queue down, set the score to exact score
            elif h > marking.h:
                self.lp_solved += 1
                marking.f = marking.g + h
                marking.h = h
                marking.x = deepcopy(x)
                marking.trust = True
                # need to requeue the marking
                return marking, "REQUEUED"
            else:
                self.lp_solved += 1
                # continue with this marking
                marking.f = marking.g + h
                marking.h = h
                marking.x = deepcopy(x)
                marking.trust = trustable
        return marking, "CLOSEDSUCCESSFUL"

    def expand_marking(self, cost_function, curr, incidence_matrix, cost_vec, closed):
        # get subsequent firsing transitions
        enabled_trans = set()
        for p in curr.m:
            for t in p.ass_trans:
                if t.sub_marking <= curr.m:
                    enabled_trans.add(t)
        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans]
        # firing subsequent transitions to get new marking
        start_time = timeit.default_timer()
        enabled_trans = sorted(sorted(trans_to_visit_with_cost, key=lambda k: str(k[0].name[1])),
                               key=lambda k: k[1])
        self.time_sort += timeit.default_timer() - start_time
        for t, cost in enabled_trans:
            # compute the new g score of the subsequent marking reached if t would be fired
            new_g = curr.g + cost
            subseq_marking = utils.add_markings(curr.m, t.add_marking)
            self.traversed += 1
            self.queued += 1
            # subsequent marking is fresh, compute the f score of this path and add it to open set
            if subseq_marking not in closed:
                if not self.open_set.heap_find(subseq_marking):
                    self.order += 1
                    tp = Marking(new_g, new_g, 0, subseq_marking, curr, t, None, False, self.order)
                    update_tp = self.derive_or_estimate_heuristic(curr, tp, incidence_matrix, cost_vec, t)
                    start_time = timeit.default_timer()
                    self.open_set.heap_insert(update_tp)
                    if timeit.default_timer() - start_time > 0.005:
                        # print("p5")
                        self.heap_longtime += 1
                    self.time_heap += timeit.default_timer() - start_time
                    self.heap_count += 1
                    self.heap_insert += 1
                # subsequent marking has shorter path
                elif new_g < self.open_set.heap_get(subseq_marking).g:
                    start_time = timeit.default_timer()
                    i = self.open_set.heap_get(subseq_marking)
                    if timeit.default_timer() - start_time > 0.005:
                        # print("p6")
                        self.heap_longtime += 1

                    self.time_heap += timeit.default_timer() - start_time
                    self.heap_count += 1
                    self.heap_retrieval += 1
                    i.g = new_g
                    i.t = t
                    i.p = curr
                    temp_h = i.h
                    if not i.trust:
                        i = self.derive_or_estimate_heuristic(curr, i, incidence_matrix, cost_vec, t)
                    start_time = timeit.default_timer()
                    self.open_set.heap_remove(i.m)
                    if timeit.default_timer() - start_time > 0.005:
                        # print("p7_1")
                        self.heap_longtime += 1
                    self.time_heap += timeit.default_timer() - start_time

                    start_time = timeit.default_timer()

                    self.open_set.heap_insert(i)
                    if timeit.default_timer() - start_time > 0.005:
                        # print("p7_2")
                        self.heap_longtime += 1

                    self.time_heap += timeit.default_timer() - start_time
                    self.heap_count += 1
                    self.heap_delete += 1
                    self.heap_insert += 1
                # subsequent marking has longer or equal path, but the heuristic change from invalid to valid
                else:
                    start_time = timeit.default_timer()
                    i = self.open_set.heap_get(subseq_marking)
                    if timeit.default_timer() - start_time > 0.005:
                        self.heap_longtime += 1
                        # print("p8")
                    self.time_heap += timeit.default_timer() - start_time
                    self.heap_count += 1
                    self.heap_retrieval += 1
                    if not i.trust:
                        i = self.derive_or_estimate_heuristic(curr, i, incidence_matrix, cost_vec, t)
                        start_time = timeit.default_timer()
                        self.open_set.heap_remove(i.m)
                        self.open_set.heap_insert(i)
                        if timeit.default_timer() - start_time > 0.005:
                            self.heap_longtime += 1
                            # print("p9")
                        self.time_heap += timeit.default_timer() - start_time
                        self.heap_delete += 1
                        self.heap_insert += 1
                        self.heap_count += 1

    def derive_or_estimate_heuristic(self, from_marking, to_marking, incidence_matrix, cost_vec, t):
        # if from marking has exact heuristic, we can derive from it
        if from_marking.trust and from_marking.x[incidence_matrix.transitions[t]] >= 1 and from_marking.h != "HEURISTICINFINITE":
            x_prime = deepcopy(from_marking.x)
            x_prime[incidence_matrix.transitions[t]] -= 1
            to_marking.x = x_prime
            to_marking.h = from_marking.h - cost_vec[incidence_matrix.transitions[t]]
            to_marking.f = to_marking.g + to_marking.h
            to_marking.trust = True
            if t.label[0] != ">>":
                if self.get_max_events(to_marking) > self.max_rank:
                    self.max_rank = self.get_max_events(to_marking)
        # if heuristic of from marking is infinite, then we return
        elif from_marking.h == "HEURISTICINFINITE":
            to_marking.h = "HEURISTICINFINITE"
            to_marking.trust = True
            x_prime = deepcopy(from_marking.x)
            x_prime[incidence_matrix.transitions[t]] -= 1
            to_marking.x = x_prime
        else:
            if to_marking.m == self.final_marking:
                to_marking.h = 0
                to_marking.f = to_marking.g
                to_marking.trust = True
            else:
                h = from_marking.h - cost_vec[incidence_matrix.transitions[t]]
                if h < 0:
                    h = 0
                if h > to_marking.h:
                    to_marking.h = h
                    to_marking.f = h + to_marking.g
                    to_marking.trust = False
        return to_marking

    def get_max_events(self, marking):
        if marking.t is None:
            return -1
        elif marking.t.label[0] != ">>":
            return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())
        else:
            return self.get_max_events(marking.p)

    def get_path_length(self, marking):
        if marking.p is None:
            return 0
        else:
            return 1 + self.get_path_length(marking.p)

    def reconstruct_alignment(self, state, trace_length, ret_tuple_as_trans_desc=False):
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
        return {
            'cost': state.g,
            'lp_solved': self.lp_solved,
            'restart': self.restart,
            "time_h": self.time_h,
            "time_heap": self.time_heap,
            'visited_states': self.visited,
            'queued_states': self.queued,
            'traversed_arcs': self.traversed,
            'trace_length': trace_length,
            'alignment': alignment,
        }

    def trust_solution(self, x):
        for v in x:
            if v < 0:
                return False
        return True

    def vectorize_initial_final_cost(self, incidence_matrix, ini, fin, cost_function):
        ini_vec = incidence_matrix.encode_marking(ini)
        fin_vec = incidence_matrix.encode_marking(fin)
        cost_vec = [0] * len(cost_function)
        for t in cost_function.keys():
            cost_vec[incidence_matrix.transitions[t]] = cost_function[t]
        return np.array(ini_vec), np.array(fin_vec), np.array(cost_vec)

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


class Marking:
    def __init__(self, f, g, h, m, p, t, x, trust, order):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        self.order = order

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        if self.trust != other.trust:
            return self.trust
        max_event1 = self.get_max_events(self)
        max_event2 = self.get_max_events(other)
        if max_event1 != max_event2:
            return max_event1 > max_event2
        if self.g > other.g:
            return True
        elif self.g < other.g:
            return False
        path1 = self.get_path_length(self)
        path2 = self.get_path_length(other)
        if path1 != path2:
            return path1 > path2
        return self.order < other.order

    def get_max_events(self, marking):
        if marking.t is None:
            return -1
        if marking.t.label[0] != ">>":
            return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())
        return self.get_max_events(marking.p)

    def get_path_length(self, marking):
        if marking.p is None:
            return 0
        else:
            return 1 + self.get_path_length(marking.p)

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


def _has_parent(idx):
    if idx == 0:
        return False
    else:
        return True

class MinHeap:
    def __init__(self):
        self.lst = []
        self.idx = {}

    def swap(self, idx1, idx2):
        # temp = self.lst[idx1]
        # self.lst[idx1] = self.lst[idx2]
        # self.idx[self.lst[idx1].m] = idx1
        # self.lst[idx2] = temp
        # self.idx[self.lst[idx2].m] = idx2
        # swap two elements for the index map and list position
        self.lst[idx1], self.lst[idx2] = self.lst[idx2], self.lst[idx1]
        self.idx[self.lst[idx1].m], self.idx[self.lst[idx2].m] = self.idx[self.lst[idx2].m],  self.idx[self.lst[idx1].m]

    def heap_insert(self, marking):
        self.lst.append(marking)
        self.idx[marking.m] = len(self.lst)-1
        self._heap_heapify_up()

    def heap_pop(self):
        # print(len(self.lst), self.lst[0])
        marking_to_pop = self.lst[0]
        del self.idx[marking_to_pop.m]
        # update list and index
        self.lst[0] = self.lst[-1]
        self.idx[self.lst[0].m] = 0
        # remove the last element
        self.lst.pop()
        self._heap_heapify_down(0)
        return marking_to_pop

    def heap_find(self, m):
        return True if m in self.idx else False

    def heap_get(self, m):
        return self.lst[self.idx[m]]

    def heap_remove(self,m):
        idx_to_remove = self.idx[m]
        marking_to_pop = self.lst[self.idx[m]]
        if len(self.lst)-1 == 1:
            self.heap_clear()
        else:
            del self.idx[marking_to_pop.m]
            # update list and index
            self.idx[self.lst[-1].m] = idx_to_remove
            self.lst[idx_to_remove] = self.lst[-1]
            # remove the last element
            self.lst.pop()
            self._heap_heapify_down(idx_to_remove)

    def heap_clear(self):
        self.lst.clear()
        self.idx.clear()

    def _heap_heapify_down(self, idx):
        while self._has_left_child(idx):
            smaller_index = idx*2+1
            if idx*2+2 < len(self.lst) and self.lst[idx*2+2] < self.lst[idx*2+1]:
                smaller_index = idx*2+2
            if self.lst[idx] < self.lst[smaller_index]:
                break
            else:
                self.swap(smaller_index, idx)
            idx = smaller_index

    def _heap_heapify_up(self):
        # start_time = timeit.default_timer()
        index = len(self.lst) - 1
        # flag = 0
        while index != 0 and self.lst[index] < self.lst[int((index - 1) // 2)]:
            # self.swap(int((index - 1) // 2), index)
            parent_index = int((index - 1) // 2)
            self.lst[index], self.lst[parent_index] = self.lst[parent_index], self.lst[index]
            self.idx[self.lst[index].m], self.idx[self.lst[parent_index].m] = \
                self.idx[self.lst[parent_index].m], self.idx[self.lst[index].m]
            index = parent_index

    def _has_left_child(self, idx):
        if idx*2+1 < len(self.lst):
            return True
        else:
            return False

    def _get_parent(self, index):
        return int((index - 1) // 2)

    def _has_right_child(self, idx):
        if idx*2+2 < len(self.lst):
            return True
        else:
            return False

    def print_idx(self):
        print(self.idx)

    def print_lst(self):
        for i in self.lst:
            print(i.m)

    def get_len(self):
        return len(self.lst)