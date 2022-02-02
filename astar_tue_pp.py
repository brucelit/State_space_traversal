import heapq

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
import time
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star
from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.lp import solver as lp_solver
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from typing import Optional, Dict, Any, Union, Tuple
from pm4py.objects.log.obj import EventLog, EventStream, Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing
import pandas as pd


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
        self.lp_for_ini_solved = 0
        self.max_rank = -2
        self.time_h = 0
        self.open_set = minheap.MinHeap()
        self.time_heap = 0
        self.time_sort = 0
        self.order = 0
        self.split_lst = []
        self.cache_set = []
        self.sync_prod = None
        self.incidence_matrix = None
        self.normal_astar_flag = False
        self.counter = 0

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
            res = {'cost': 0, 'lp_solved': 0, 'restart': 0, "time_h": 0, "time_heap": 0, 'visited_states': 0,
                   'queued_states': 0, 'traversed_arcs': 0, 'trace_length': len(trace_lst), 'alignment': None, 'time_sum': 0,
                   'time_diff': 0}
        else:
            res['time_sum'] = timeit.default_timer() - start_time
            res['time_diff'] = res['time_sum'] - res['time_h']
        return res

    def search(self, ini, fin, cost_function, incidence_matrix, trace_sync, trace_log):
        ini_vec, fin_vec, cost_vec = self.vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)
        closed = {}
        p_index = incidence_matrix.places
        # print("t index", incidence_matrix.transitions)
        inc_matrix = incidence_matrix.a_matrix
        cons_matrix = incidence_matrix.b_matrix
        search_start_time = timeit.default_timer()

        start_time = timeit.default_timer()
        h, x = get_exact_heuristic(fin_vec - ini_vec, inc_matrix, cost_vec)
        self.time_h += timeit.default_timer() - start_time
        self.lp_solved += 1

        # add initial marking to open set
        ini_state = Marking(h, 0, h, ini, None, None, [deepcopy(x)], True, self.order, [[]])
        start_time = timeit.default_timer()
        self.open_set.heap_insert(ini_state)
        self.time_heap += timeit.default_timer() - start_time

        # while not all states visited
        while self.open_set:
            # if (timeit.default_timer() - search_start_time) > 1000:
            #     return None
            # get the most promising marking
            start_time = timeit.default_timer()
            new_curr = self.open_set.heap_pop()
            self.time_heap += timeit.default_timer() - start_time

            # if new_curr.f != new_curr.g + new_curr.h:
            #     print("不一致", new_curr.f, new_curr.g, new_curr.h)
            # if self.visited % 4000 == 0:
            #     print("\n", self.visited, new_curr.m, "f:", new_curr.f, "g:", new_curr.g, "h:", new_curr.h,
            #           new_curr.trust)
            if len(self.split_lst) == 3 and new_curr.t is not None:
                print(self.visited, new_curr.m, "paths len:", len(new_curr.pre_trans_lst), "open set len: ", len(self.open_set.lst),
                      "f:", new_curr.f, "g:", new_curr.g, "h:", new_curr.h,
                      new_curr.trust, new_curr.p.m, new_curr.p.f, new_curr.p.g)
            # print("\n", self.visited, new_curr.m, new_curr.trust, new_curr.f, new_curr.g, new_curr.pre_trans_lst)
            marking_diff = fin_vec - incidence_matrix.encode_marking(new_curr.m)
            curr, flag = \
                self.close_or_update_marking(new_curr, ini_state.m, fin, cost_vec,
                                             marking_diff, incidence_matrix, len(trace_log))
            if flag == "CLOSEDSUCCESSFUL":
                closed[curr.m] = curr
                self.visited += 1
                self.expand_marking(cost_function, curr, incidence_matrix, cost_vec, closed, h, deepcopy(x))
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
                self.lp_for_ini_solved += 1
                self.max_rank = -2

                # add to cache set
                cache_set = []
                valid_flag = False
                for each_marking in self.open_set.lst:
                    # update each marking and add to cache set
                    new_marking = self.update_marking(each_marking, x, cost_vec, h)
                    cache_set.append(new_marking)
                # update each marking and add to cache set
                new_curr_marking = self.update_marking(curr, x, cost_vec, h)
                cache_set.append(new_curr_marking)
                # put marking in cache set into open set
                for new_marking in cache_set:
                    if new_marking.trust:
                        valid_flag = True
                # if no markings in cache set become valid, then we fall back to normal A*
                if not valid_flag:
                    print("\nfall back to original extended A*")
                    self.restart += 1
                    closed = {}
                    self.order = 0
                    ini_state = Marking(h, 0, h, ini, None, None, [deepcopy(x)], True, self.order, [[]])
                    start_time = timeit.default_timer()
                    self.open_set = minheap.MinHeap()
                    self.open_set.heap_insert(ini_state)
                    self.time_heap += timeit.default_timer() - start_time
                    self.max_rank = -2
                else:
                    self.open_set.heap_clear()
                    print("check is ok", len(self.open_set.lst))
                    for new_marking in cache_set:
                        self.open_set.heap_insert(new_marking)
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

            if self.visited == 189:
                print("check open set:")
                for i in self.open_set.lst:
                    print(i.m, i.trust, i.f, i.g, i.pre_trans_lst)

            start_time = timeit.default_timer()
            h, x, trustable, self.split_lst, self.max_rank = \
                get_exact_heuristic_new(marking, self.split_lst, marking_diff, ini, incidence_matrix.a_matrix,
                                        cost_vec, self.max_rank, len_trace)

            self.time_h += timeit.default_timer() - start_time
            print("lp solved", self.lp_solved)
            # need to restart
            if h == -1:
                return marking, "RESTARTNEEDED"
            # heuristic is not computable, from which final marking is unreachable
            elif trustable == "Infeasible":
                self.lp_solved += 1
                return marking, "CLOSEDINFEASIBLE"
            else:
                self.lp_solved += 1
                # continue with this marking
                marking.f = marking.g + h
                marking.h = h
                marking.x = [deepcopy(x)]
                marking.trust = trustable
                self.queued += 1
                return marking, "REQUEUED"
        return marking, "CLOSEDSUCCESSFUL"

    def expand_marking(self, cost_function, curr, incidence_matrix, cost_vec, closed, ini_h, ini_x):
        # get subsequent firing transitions
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
        self.time_sort += timeit.default_timer() - start_time
        if len(self.split_lst) == 3 and self.visited > 185:
            print("enabled trans", len(enabled_trans), enabled_trans)
        for t, cost in enabled_trans:
            new_g = curr.g + cost
            new_m = utils.add_markings(curr.m, t.add_marking)
            self.traversed += 1
            # subsequent marking is not fresh, compute the f score of this path and add it to open set
            if new_m in closed:
                i = closed[new_m]
                pre_trans = deepcopy(curr.pre_trans_lst)
                # if len(self.split_lst) == 7:
                #     print("update closed before: \n", pre_trans, "\n", i.pre_trans_lst)
                if new_g <= i.g:
                    for j in pre_trans:
                        j.append(incidence_matrix.transitions[t])
                    i, fl = self.update_paths(pre_trans, i)

                    if fl:
                        new_i = self.update_marking(i, ini_x, cost_vec, ini_h)
                        if len(self.split_lst) == 3:
                            print("need to update this one", new_i.m)

                        del closed[new_m]
                        start_time = timeit.default_timer()
                        self.open_set.heap_insert(new_i)
                        self.time_heap += timeit.default_timer() - start_time
                        self.queued += 1
            else:
                if not self.open_set.heap_find(new_m):
                    # add new paths
                    pre_trans = deepcopy(curr.pre_trans_lst)
                    for j in pre_trans:
                        j.append(incidence_matrix.transitions[t])
                    self.queued += 1
                    self.order += 1
                    tp = Marking(new_g, new_g, 0, new_m, curr, t, None, False, self.order, deepcopy(pre_trans))
                    update_tp = self.derive_or_estimate_heuristic(curr, tp, incidence_matrix, cost_vec, t)
                    start_time = timeit.default_timer()
                    self.open_set.heap_insert(update_tp)
                    self.time_heap += timeit.default_timer() - start_time

                # subsequent marking has shorter path
                elif new_g < self.open_set.heap_get(new_m).g:
                    start_time = timeit.default_timer()
                    i = self.open_set.heap_get(new_m)
                    self.time_heap += timeit.default_timer() - start_time
                    pre_trans = deepcopy(curr.pre_trans_lst)
                    for j in pre_trans:
                        j.append(incidence_matrix.transitions[t])
                    new_i = Marking(i.f, i.g, i.h, i.m, i.p, i.t, deepcopy(i.x), i.trust, i.order, deepcopy(i.pre_trans_lst))
                    new_i, fl = self.update_paths(pre_trans, new_i)
                    new_i.g = new_g
                    new_i.t = t
                    new_i.p = curr
                    self.order += 1
                    new_i.order = self.order
                    if not new_i.trust:
                        new_i = self.derive_or_estimate_heuristic(curr, new_i, incidence_matrix, cost_vec, t)
                    new_i.f = new_i.g + new_i.h
                    i = new_i
                    start_time = timeit.default_timer()
                    self.open_set.heap_update(i)
                    self.time_heap += timeit.default_timer() - start_time

                # subsequent marking has equal path, but the heuristic change from invalid to valid
                elif new_g == self.open_set.heap_get(new_m).g:
                    start_time = timeit.default_timer()
                    i = self.open_set.heap_get(new_m)
                    self.time_heap += timeit.default_timer() - start_time
                    pre_trans = deepcopy(curr.pre_trans_lst)
                    for j in pre_trans:
                        j.append(incidence_matrix.transitions[t])
                    i, fl = self.update_paths(pre_trans, i)
                    new_i = Marking(i.f, i.g, i.h, i.m, i.p, i.t, deepcopy(i.x), i.trust, i.order, deepcopy(i.pre_trans_lst))
                    if not new_i.trust:
                        new_i = self.derive_or_estimate_heuristic(curr, new_i, incidence_matrix, cost_vec, t)
                        if new_i.trust:
                            new_i.t = t
                            new_i.p = curr
                            self.order += 1
                            new_i.order = self.order
                            i = new_i
                            start_time = timeit.default_timer()
                            self.open_set.heap_update(i)
                            self.time_heap += timeit.default_timer() - start_time
                    else:
                        start_time = timeit.default_timer()
                        self.open_set.heap_update(new_i)
                        self.time_heap += timeit.default_timer() - start_time

                # subsequent marking has longer path, but the heuristic change from invalid to valid
                else:
                    start_time = timeit.default_timer()
                    i = self.open_set.heap_get(new_m)
                    self.time_heap += timeit.default_timer() - start_time

                    pre_trans = deepcopy(curr.pre_trans_lst)
                    for j in pre_trans:
                        j.append(incidence_matrix.transitions[t])
                    # if len(self.split_lst) == 3 and self.visited > 185:
                    #     print("start checking paths:")
                    #     start_time = timeit.default_timer()
                    # i, fl = self.update_paths(pre_trans, i)
                    # if len(self.split_lst) == 3 and self.visited > 185:
                    #     print("checking paths finished:", timeit.default_timer() - start_time)
                    #
                    # if len(self.split_lst) == 3 and self.visited > 185:
                    #     print("found in longer g:", i.m, len(i.pre_trans_lst))
                    new_i = Marking(i.f, i.g, i.h, i.m, i.p, i.t, deepcopy(i.x), i.trust, i.order, deepcopy(i.pre_trans_lst))
                    # if fl:
                    #     print("add new paths", new_i.pre_trans_lst)
                    if not i.trust:
                        new_i = self.derive_or_estimate_heuristic(curr, new_i, incidence_matrix, cost_vec, t)
                        # print("expand marking 4: ", new_i.m, new_i.trust, new_i.f, new_i.g, new_i.h, new_i.pre_trans_lst)
                        if new_i.trust:
                            start_time = timeit.default_timer()
                            self.open_set.heap_update(new_i)
                            self.time_heap += timeit.default_timer() - start_time
                    else:
                        start_time = timeit.default_timer()
                        self.open_set.heap_update(new_i)
                        self.time_heap += timeit.default_timer() - start_time

    def derive_or_estimate_heuristic(self, from_marking, to_marking, incidence_matrix, cost_vec, t):
        # assume we could not derive valid sol and heuristic from from_marking
        trust_flag = False

        # use mark list to check whether there are paths that have valid solution vec
        valid_sol_mark = [1 for i in range(len(from_marking.x))]

        # use h list to store only valid h
        # use x list to store only valid solution vec
        valid_sol_lst = []
        valid_h_lst = []

        # use h_to_add_lst to store all possible h
        # use solution_to_add_lst to store all possible solution vec
        solution_to_add_lst = []
        h_to_add_lst = []

        count = 0
        for each_sol in from_marking.x:
            solution_vec = deepcopy(each_sol)
            temp_h = from_marking.h
            solution_vec[incidence_matrix.transitions[t]] -= 1
            temp_h -= cost_vec[incidence_matrix.transitions[t]]
            # when the solution vector encounters <0, means no longer trustable
            if solution_vec[incidence_matrix.transitions[t]] < 0:
                valid_sol_mark[count] = 0
            h_to_add_lst.append(temp_h)
            solution_to_add_lst.append(solution_vec)
            count += 1

        if 1 in valid_sol_mark:
            trust_flag = True
            for i in range(len(valid_sol_mark)):
                if valid_sol_mark[i] == 1:
                    valid_sol_lst.append(solution_to_add_lst[i])
                    valid_h_lst.append(h_to_add_lst[i])
            if t.label[0] != ">>":
                if self.get_max_events(from_marking) + 1 > self.max_rank:
                    self.max_rank = self.get_max_events(from_marking) + 1
                    # if len(self.split_lst) == 2:
                    #     print("from marking", from_marking.m, self.max_rank)

        if trust_flag:
            # print("trust", to_marking.m, h_to_add_lst)
            to_marking.h = max(0, max(valid_h_lst))
            to_marking.x = deepcopy(valid_sol_lst)
        else:
            # print("not trust", to_marking.m, h_to_add_lst)
            to_marking.h = max(max(h_to_add_lst), 0)
            to_marking.x = [[]]

        to_marking.f = to_marking.g + to_marking.h
        to_marking.trust = trust_flag
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

    # when new paths could be added to the pre_trans_lst
    def update_paths(self, pre_trans, to_marking):
        # use mark to check whether pre_trans has new paths
        mark = [1 for i in range(len(pre_trans))]
        flag = False
        count = 0
        for j in pre_trans:
            for k in to_marking.pre_trans_lst:
                if (len(j) == len(k) and set(j) == set(k)) or set(k) == set(j[0:len(k)]):
                    mark[count] = 0
                    continue
            count += 1
        for ele in range(len(mark)):
            if mark[ele] == 1:
                flag = True
                # print("paths to append", pre_trans[ele])
                to_marking.pre_trans_lst.append(pre_trans[ele])
        return to_marking, flag

    # compute solutions for the state
    def update_marking(self, marking, new_sol, cost_vec, h):
        # get the new solution vec from initial marking
        trust_flag = False

        # use mark list to check whether there are paths that have valid solution vec
        valid_path_mark = [1 for i in range(len(marking.pre_trans_lst))]

        # use h list to store only valid h
        # use x list to store only valid solution vec
        valid_sol_lst = []
        valid_h_lst = []

        # use h_to_add_lst to store all possible h
        # use solution_to_add_lst to store all possible solution vec
        solution_to_add_lst = []
        h_to_add_lst = []

        count = 0
        for trans_lst in marking.pre_trans_lst:
            solution_vec = deepcopy(new_sol)
            temp_h = h
            for trans in trans_lst:
                solution_vec[trans] -= 1
                temp_h -= cost_vec[trans]
                # when the solution vector encounters <0, means no longer trustable
                if solution_vec[trans] < 0:
                    valid_path_mark[count] = 0
            h_to_add_lst.append(temp_h)
            solution_to_add_lst.append(solution_vec)
            count += 1

        if 1 in valid_path_mark:
            trust_flag = True
            for i in range(len(valid_path_mark)):
                if valid_path_mark[i] == 1:
                    valid_sol_lst.append(solution_to_add_lst[i])
                    valid_h_lst.append(h_to_add_lst[i])

        if trust_flag:
            marking.h = max(0, min(valid_h_lst))
            marking.x = deepcopy(valid_sol_lst)
            # if len(self.split_lst) == 7:
            #     print("valid h:", marking.m, marking.h, h_to_add_lst)
        else:
            marking.h = max(0, max(h_to_add_lst))
            marking.x = [[]]
            # if len(self.split_lst) == 7:
            #     print("invalid h:", marking.m, marking.h, h_to_add_lst)
        marking.f = marking.g + marking.h
        marking.trust = trust_flag
        return marking

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
    def __init__(self, f, g, h, m, p, t, x, trust, order, pre_trans_lst):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        self.order = order
        self.pre_trans_lst = pre_trans_lst

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
