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
        self.lp_for_ini_solved = 0
        self.restart = 0
        self.max_rank = -2
        self.time_h = 0
        self.open_set = minheap.MinHeap()
        self.time_heap = 0
        self.time_sort = 0
        self.order = 0
        self.split_lst = []
        self.sync_prod = None
        self.incidence_matrix = None

    def apply(self, trace, petri_net, initial_marking, final_marking, split_lst, parameters=None):
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
        self.split_lst = split_lst
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

        sync_prod, sync_initial_marking, sync_final_marking, cost_function = construction.construct_cost_aware(
            trace_net, trace_im, trace_fm, petri_net, initial_marking, final_marking, utils.SKIP,
            trace_net_costs, model_cost_function, revised_sync)
        self.final_marking = sync_final_marking
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
        self.sync_prod = sync_prod
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
        self.incidence_matrix = incidence_matrix
        res = self.search(initial_marking, final_marking, cost_function, incidence_matrix,
                          trace_sync, trace_log)
        if res is None:
            res = {}
            res['cost'] = 0
            res['lp_solved'] = 0
            res['restart'] = 0
            res["time_h"] = 0
            res["time_heap"] = 0
            res['visited_states'] = 0
            res['queued_states'] = 0
            res['traversed_arcs'] = 0
            res['trace_length'] = 0
            res['alignment'] = None
            res['time_sum'] = 0
            res['time_diff'] = 0
        else:
            res['time_sum'] = timeit.default_timer() - start_time
            res['time_diff'] = res['time_sum'] - res['time_h']
        return res

    def search(self, ini, fin, cost_function, incidence_matrix, trace_sync, trace_log):
        ini_vec, fin_vec, cost_vec = self.vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)
        closed = set()
        p_index = incidence_matrix.places
        inc_matrix = incidence_matrix.a_matrix
        cons_matrix = incidence_matrix.b_matrix
        search_start_time = timeit.default_timer()

        start_time = timeit.default_timer()

        # h, x = get_exact_heuristic(fin_vec - ini_vec, inc_matrix, cost_vec)

        print("precompute split_lst:", self.split_lst)
        h, x = get_ini_heuristic(ini_vec, fin_vec, cost_vec, sorted(self.split_lst), inc_matrix, cons_matrix,
                                 incidence_matrix.transitions, p_index,
                                 trace_sync, trace_log)
        # h, x = get_exact_heuristic(fin_vec-incidence_matrix.encode_marking(ini), inc_matrix, cost_vec)
        self.time_h += timeit.default_timer() - start_time
        self.lp_solved += 1

        # add initial marking to open set
        ini_state = Marking(h, 0, h, ini, None, None, deepcopy(x), True, self.order)
        start_time = timeit.default_timer()
        self.open_set.heap_insert(ini_state)
        self.time_heap += timeit.default_timer() - start_time

        # while not all states visited
        while self.open_set:
            if (timeit.default_timer() - search_start_time) > 500:
                return None

            # get the most promising marking
            start_time = timeit.default_timer()
            new_curr = self.open_set.heap_pop()
            if new_curr.f != new_curr.g + new_curr.h:
                print("不一致")
            # if self.visited % 2000 == 0:
            #     print("\n", self.visited, new_curr.m, "f:", new_curr.f, "g:", new_curr.g, "h:", new_curr.h, new_curr.trust)
            self.time_heap += timeit.default_timer() - start_time
            marking_diff = fin_vec - incidence_matrix.encode_marking(new_curr.m)
            curr, flag = \
                self.close_or_update_marking(new_curr, ini_state.m, fin, cost_vec,
                                             marking_diff, incidence_matrix, len(trace_log))
            if flag == "CLOSEDSUCCESSFUL":
                closed.add(curr.m)
                self.visited += 1
                self.expand_marking(cost_function, curr, incidence_matrix, cost_vec, closed)
            elif flag == "REQUEUED":
                start_time = timeit.default_timer()
                self.open_set.heap_insert(curr)
                self.time_heap += timeit.default_timer() - start_time
            elif flag == "RESTARTNEEDED":
                self.split_lst = sorted(self.split_lst)
                start_time = timeit.default_timer()
                print("split_lst", self.split_lst)
                h, x = get_ini_heuristic(ini_vec, fin_vec, cost_vec, self.split_lst, inc_matrix, cons_matrix,
                                         incidence_matrix.transitions, p_index,
                                         trace_sync, trace_log)
                self.time_h += timeit.default_timer() - start_time
                self.lp_for_ini_solved = 0
                self.restart += 1
                # restart by reset open set and closed set
                closed = set()
                self.order = 0
                ini_state = Marking(h, 0, h, ini, None, None, deepcopy(x), True, self.order)
                start_time = timeit.default_timer()
                self.open_set = minheap.MinHeap()
                self.open_set.heap_insert(ini_state)
                self.time_heap += timeit.default_timer() - start_time
                self.max_rank = -2
            elif flag == "FINALMARKINGFOUND":
                return self.reconstruct_alignment(curr, len(trace_log))
            elif flag == "CLOSEDINFEASIBLE":
                closed[curr.m] = curr

    def close_or_update_marking(self, marking, ini, fin, cost_vec, marking_diff,
                                incidence_matrix, len_trace):
        if marking.m == fin and marking.h == 0:
            return marking, "FINALMARKINGFOUND"
        # if the heuristic is not exact
        if not marking.trust:
            # compute the exact heuristics
            self.split_lst = sorted(self.split_lst)
            start_time = timeit.default_timer()
            # get_exact_heuristic_new(marking, split_lst, fin_vec, ini, incidence_matrix, cost_vec, max_rank, trace_len, trans_len, sync_net)
            h, x, trustable, self.split_lst, self.max_rank = \
                get_exact_heuristic_new(marking, self.split_lst, marking_diff, ini, incidence_matrix.a_matrix,
                                        cost_vec, self.max_rank, len_trace)
            print("lp solved", self.lp_solved)
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
                # if len(self.split_lst) == 7:
                #     print("after computation", marking.m, "f:", marking.f, "g", marking.g, "h:", marking.h)
                self.queued += 1
                # need to requeue the marking
                return marking, "REQUEUED"
            else:
                self.lp_solved += 1
                # continue with this marking
                marking.f = marking.g + h
                marking.h = h
                marking.x = deepcopy(x)
                marking.trust = trustable
                # print("compute exact h", marking.m, "original h:", marking.h, "new h", h, trustable, "f:", marking.f,
                #       "g:", marking.g)
        return marking, "CLOSEDSUCCESSFUL"

    def expand_marking(self, cost_function, curr, incidence_matrix, cost_vec, closed):
        # get subsequent firsing transitions
        enabled_trans = set()
        for p in curr.m:
            for t in p.ass_trans:
                if t.sub_marking <= curr.m:
                    enabled_trans.add(t)
        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and is_log_move(t, utils.SKIP) and is_model_move(t, utils.SKIP))]
        # firing subsequent transitions to get new marking
        start_time = timeit.default_timer()
        enabled_trans = sorted(sorted(trans_to_visit_with_cost, key=lambda k: str(k[0].name[0])),
                               key=lambda k: k[1], reverse=True)
        # if len(self.split_lst) == 9:
        #     print("enabled transitions:", len(trans_to_visit_with_cost), trans_to_visit_with_cost)
        self.time_sort += timeit.default_timer() - start_time
        for t, cost in enabled_trans:
            # compute the new g score of the subsequent marking reached if t would be fired
            new_g = curr.g + cost
            new_m = utils.add_markings(curr.m, t.add_marking)
            self.traversed += 1
            # subsequent marking is fresh, compute the f score of this path and add it to open set
            if new_m not in closed:
                if not self.open_set.heap_find(new_m):
                    self.queued += 1
                    self.order += 1
                    tp = Marking(new_g, new_g, 0, new_m, curr, t, None, False, self.order)
                    update_tp = self.derive_or_estimate_heuristic(curr, tp, incidence_matrix, cost_vec, t)
                    start_time = timeit.default_timer()
                    self.open_set.heap_insert(update_tp)
                    self.time_heap += timeit.default_timer() - start_time
                    # if len(self.split_lst) == 9:
                    #     print("expand marking:", tp.m, tp.f, tp.f, "g:", tp.g, "h:", tp.h, tp.trust, t)
                # subsequent marking has shorter path
                elif new_g < self.open_set.heap_get(new_m).g:
                    start_time = timeit.default_timer()
                    i = self.open_set.heap_get(new_m)
                    self.time_heap += timeit.default_timer() - start_time
                    # print("update marking 1 before:", i.m, "f:", i.f, "g:", i.g, "h:", i.h, i.trust, i.t, t)
                    i.g = new_g
                    i.t = t
                    i.p = curr
                    if not i.trust:
                        i = self.derive_or_estimate_heuristic(curr, i, incidence_matrix, cost_vec, t)
                    i.f = i.g + i.h
                    # print("update marking 1 after:", i.m, "f:", i.f, "g:", i.g, "h:", i.h, i.trust, i.t, t)
                    start_time = timeit.default_timer()
                    self.open_set.heap_update(i)
                    self.time_heap += timeit.default_timer() - start_time

                # subsequent marking has longer or equal path, but the heuristic change from invalid to valid
                elif new_g == self.open_set.heap_get(new_m).g:
                    start_time = timeit.default_timer()
                    i = self.open_set.heap_get(new_m)
                    self.time_heap += timeit.default_timer() - start_time
                    # print("update marking 2 before:", i.m, "f:", i.f, "g:", i.g, "h:", i.h, i.trust,i.t, t)
                    if not i.trust:
                        new_i = self.derive_or_estimate_heuristic(curr, i, incidence_matrix, cost_vec, t)
                        if new_i.trust:
                            new_i.t = t
                            new_i.p = curr
                            start_time = timeit.default_timer()
                            self.open_set.heap_update(new_i)
                            self.time_heap += timeit.default_timer() - start_time
                            # print("update marking 2 after:",  i.m, "f:", i.f, "g:", i.g, "h:", i.h, i.trust, i.t, t)
                else:
                    start_time = timeit.default_timer()
                    i = self.open_set.heap_get(new_m)
                    self.time_heap += timeit.default_timer() - start_time
                    # print("update marking 3 before:", i.m, "f:", i.f, "g:", i.g, "h:", i.h, i.trust, i.t, t)
                    if not i.trust:
                        new_i = self.derive_or_estimate_heuristic(curr, i, incidence_matrix, cost_vec, t)
                        if new_i.trust:
                            start_time = timeit.default_timer()
                            self.open_set.heap_update(new_i)
                            self.time_heap += timeit.default_timer() - start_time
                            # print("update marking 3 after:", i.m, "f:", i.f, "g:", i.g, "h:", i.h, i.trust, i.t, t)

    def derive_or_estimate_heuristic(self, from_marking, to_marking, incidence_matrix, cost_vec, t):
        # if from marking has exact heuristic, we can derive from it
        if from_marking.x[incidence_matrix.transitions[t]] >= 1 and from_marking.h != "HEURISTICINFINITE":
            x_prime = deepcopy(from_marking.x)
            x_prime[incidence_matrix.transitions[t]] -= 1
            to_marking.x = x_prime
            to_marking.h = from_marking.h - cost_vec[incidence_matrix.transitions[t]]
            to_marking.f = to_marking.g + to_marking.h
            to_marking.trust = True
            if t.label[0] != ">>":
                if self.get_max_events(from_marking) + 1 > self.max_rank:
                    self.max_rank = self.get_max_events(from_marking) + 1

        # if heuristic of from marking is infinite, then we return
        elif from_marking.h == "HEURISTICINFINITE":
            to_marking.h = "HEURISTICINFINITE"
            to_marking.trust = True
            x_prime = deepcopy(from_marking.x)
            x_prime[incidence_matrix.transitions[t]] -= 1
            to_marking.x = x_prime
        else:
            # if to_marking.m == self.final_marking:
            #     to_marking.h =
            #     to_marking.f = to_marking.g
            #     to_marking.trust = True
            # else:
            h = from_marking.h - cost_vec[incidence_matrix.transitions[t]]
            if h < 0:
                h = 0
            if h > to_marking.h:
                to_marking.h = h
                to_marking.f = h + to_marking.g
                to_marking.trust = False
                to_marking.x = deepcopy(from_marking.x)
                to_marking.x[incidence_matrix.transitions[t]] -= 1
        return to_marking

    def get_max_events(self, marking):
        if marking.t is None:
            return -2
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
            'lp_for_ini_solved': self.lp_for_ini_solved,
            'restart': self.restart,
            'time_h': self.time_h,
            'time_heap': self.time_heap,
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
        self.__A, self.__B, self.__place_indices, self.__transition_indices, self.__rule_l, self.__rule_r = self.__construct_matrix(net)

    def encode_marking(self, marking):
        x = [0 for i in range(len(self.places))]
        for p in marking:
            x[self.places[p]] = marking[p]
        return x

    def __get_a_matrix(self):
        return self.__A

    def __get_b_matrix(self):
        return self.__B

    def __get_rule_l(self):
        return self.__rule_l

    def __get_rule_r(self):
        return self.__rule_r

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

        a_matrix = np.array([[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))])
        b_matrix = np.array([[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))])
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
            if len(rule_l[count]) == 0 or len(rule_r[count]) == 0:
                continue
            count += 1
        return a_matrix, b_matrix, new_p_index, new_t_index, rule_l, rule_r

    a_matrix = property(__get_a_matrix)
    b_matrix = property(__get_b_matrix)
    places = property(__get_place_indices)
    transitions = property(__get_transition_indices)
    rule_l = property(__get_rule_l)
    rule_r = property(__get_rule_r)

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
        return self.order > other.order

    def get_max_events(self, marking):
        if marking.t is None:
            return -2
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


def is_model_move(t, skip):
    return t.label[0] == skip and t.label[1] != skip


def is_log_move(t, skip):
    return t.label[0] != skip and t.label[1] == skip