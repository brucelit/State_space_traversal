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
        self.traversed = 0
        self.lp_solved = 0
        self.restart = 0
        self.max_rank = -1
        self.time_h = 0
        self.open_set = []
        self.cache_set = []
        self.propagation_rank = 0
        self.normal_astar_flag = False

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
        # print(res)
        return res

    def search(self, ini, fin, cost_function, incidence_matrix, trace_sync, trace_log):
        split_lst = []
        ini_vec, fin_vec, cost_vec = self.vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)
        closed = {}
        p_index = incidence_matrix.places
        inc_matrix = incidence_matrix.a_matrix
        cons_matrix = incidence_matrix.b_matrix

        # when the split list is empty, means we compute with normal LP
        start_time = timeit.default_timer()
        h, x = get_exact_heuristic(fin_vec - ini_vec, inc_matrix, cost_vec)
        self.time_h += timeit.default_timer() - start_time
        self.lp_solved += 1

        # add initial marking to open set
        ini_state = SearchTuple(h, 0, h, ini, None, None, [deepcopy(x)], True, [[]], self.propagation_rank)
        self.open_set = [ini_state]
        already_visited = {ini_state.m: ini_state}
        while 1:
        # while not all states visited
            while self.open_set:
                # get the most promising marking
                self.visited += 1
                new_curr = heapq.heappop(self.open_set)
                # if self.visited >= 2:
                #     print("\n", self.visited, new_curr.m, new_curr.trust, self.max_rank, len(self.open_set))
                marking_diff = fin_vec - incidence_matrix.encode_marking(new_curr.m)
                curr, flag, split_lst = \
                    self.close_or_update_marking(new_curr, ini_state.m, fin, cost_vec, split_lst,
                                                 marking_diff, inc_matrix, len(trace_log))

                if flag == "CLOSEDSUCCESSFUL":
                    if self.get_max_events(curr) > self.max_rank:
                        self.max_rank = self.get_max_events(curr)
                    closed[curr.m] = curr
                    # only after add a marking to closed set, can we expand from it
                    self.expand_marking(cost_function, curr, already_visited,
                                        incidence_matrix, cost_vec, closed, split_lst)
                elif flag == "REQUEUED":
                    heapq.heappush(self.open_set, curr)
                elif flag == "FINALMARKINGFOUND":
                    print(curr.pre_trans_lst)
                    return self.reconstruct_alignment(curr, len(trace_log))
                elif flag == "CLOSEDINFEASIBLE":
                    closed[curr.m] = curr
                elif flag == "Continuesearch":
                    continue

            # when the open set is blank, then we add cache to open
            print("\ntime to check, cache set len:", len(self.cache_set), "open set len:", len(self.open_set), self.max_rank)
            if self.max_rank+1 not in split_lst:
                split_lst.append(self.max_rank+1)
                splits = sorted(split_lst)
                start_time = timeit.default_timer()
                h, x = get_ini_heuristic(ini_vec, fin_vec, cost_vec, splits, inc_matrix, cons_matrix,
                                         incidence_matrix.transitions, p_index, trace_sync, trace_log)
                print("split_lst", split_lst, h)
                self.time_h += timeit.default_timer() - start_time
                self.lp_solved += 1
                self.restart += 1
                self.propagation_rank += 1

                # suppose no marking becomes valid, then we continue with normal A star in the search
                self.normal_astar_flag = True
                for i in self.cache_set:
                    update_curr = self.get_state(i, x, cost_vec, h)
                    update_curr.propagation_rank = self.propagation_rank
                    if update_curr.trust:
                        self.normal_astar_flag = False
                        if self.get_max_events(update_curr) + 1 > self.max_rank:
                            self.max_rank = self.get_max_events(update_curr) + 1
                        heapq.heappush(self.open_set, i)
                        already_visited[i.m] = copy(i)
                # if the state has in valid then keep them in cache
                self.cache_set = [i for i in self.cache_set if not i.trust]
                if self.normal_astar_flag:
                    self.open_set = self.cache_set
                    self.cache_set = []
                # print("After checking, open set:", len(self.open_set), "cache set:", len(self.cache_set))
            else:
                self.normal_astar_flag = True
                for i in self.cache_set:
                    heapq.heappush(self.open_set, i)
                self.cache_set = []
                print("open set:", len(self.open_set), len(self.cache_set))

    # visit a marking, determine whether to put into close based on the heuristic
    def close_or_update_marking(self, marking, ini, fin, cost_vec, split_lst, marking_diff,
                                incidence_matrix, len_trace):
        # if final marking reached
        if marking.m == fin and marking.trust:
            return marking, "FINALMARKINGFOUND", split_lst

        # if the heuristic is not exact
        if not marking.trust:
            # if we do not use normal astar, then we just add the marking to cache set.
            if not self.normal_astar_flag:
                self.cache_set.append(marking)
                return marking, "Continuesearch", split_lst

            # if we can no add split point, then we continue with normal a star
            else:
                split_lst = sorted(split_lst)
                start_time = timeit.default_timer()
                h, x, trustable, split_lst, self.max_rank = get_exact_heuristic_new(marking, split_lst, marking_diff,
                                                                                    ini, incidence_matrix,
                                                                                    cost_vec, self.max_rank, len_trace)
                self.time_h += timeit.default_timer() - start_time
                if self.max_rank + 1 not in split_lst:
                    print("compute exact h, the normal astar end with max rank:", self.max_rank, marking.pre_trans_lst)
                    self.normal_astar_flag = False
                    return marking, "RESTARTNEEDED", split_lst
                # need to restart
                if h == -1:
                    return marking, "RESTARTNEEDED", split_lst
                # heuristic is not computable, from which final marking is unreachable
                elif trustable == "Infeasible":
                    self.lp_solved += 1
                    return marking, "CLOSEDINFEASIBLE", split_lst
                # if the heuristic is higher push the head of the queue down, set the score to exact score
                elif h > marking.h:
                    self.lp_solved += 1
                    marking.f = marking.g + h
                    marking.h = h
                    marking.x = [deepcopy(x)]
                    marking.trust = True
                    # need to requeue the marking
                    return marking, "REQUEUED", split_lst
                else:
                    self.lp_solved += 1
                    # continue with this marking
                    marking.f = marking.g + h
                    marking.h = h
                    marking.x = [deepcopy(x)]
                    marking.trust = trustable
        return marking, "CLOSEDSUCCESSFUL", split_lst

    def expand_marking(self, cost_function, curr, already_visited, incidence_matrix, cost_vec,
                       closed, split_lst):
        # iterate over subsequent transitions
        enabled_trans = set()
        for p in curr.m:
            for t in p.ass_trans:
                if t.sub_marking <= curr.m:
                    enabled_trans.add(t)
        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans]
        for t, cost in trans_to_visit_with_cost:
            # compute the new g score of the marking reached if t would be fired
            new_g = curr.g + cost
            new_marking = utils.add_markings(curr.m, t.add_marking)
            self.traversed += 1
            # new marking is fresh, compute the f score of this path and add it to open set
            if new_marking in closed:
                # add possible new path to closed set
                state = closed[new_marking]
                if curr.propagation_rank > state.propagation_rank:
                    # print("\ncheck closed", curr.m, curr.propagation_rank, state.m, state.propagation_rank)
                    pre_trans = deepcopy(curr.pre_trans_lst)
                    for j in pre_trans:
                        j.append(incidence_matrix.transitions[t])
                    # print("before checking",state.p.g, state.p.m, state.t)
                    update_tp, flag = self.update_paths(pre_trans, state)
                    # if found new paths, continue
                    if flag:
                        update_tp = self.derive_or_estimate_heuristic(curr, update_tp, incidence_matrix, cost_vec, t)
                        # print("after checking", update_tp.p.g, update_tp.p.m, update_tp.t)
                        if new_g <= update_tp.g:
                            update_tp.g = new_g
                            update_tp.t = t
                            update_tp.p = curr
                        update_tp.f = update_tp.g + update_tp.h
                        update_tp.propagation_rank = self.propagation_rank
                        heapq.heappush(self.open_set, update_tp)
                        del closed[new_marking]
                    already_visited[new_marking] = update_tp
            else:
                if new_marking not in already_visited:
                    pre_trans = deepcopy(curr.pre_trans_lst)
                    for j in pre_trans:
                        j.append(incidence_matrix.transitions[t])
                    tp = SearchTuple(0, new_g, -1, new_marking, curr, t, [], True, deepcopy(pre_trans), curr.propagation_rank)
                    update_tp = self.derive_or_estimate_heuristic(curr, tp, incidence_matrix, cost_vec, t)
                    already_visited[new_marking] = update_tp
                    heapq.heappush(self.open_set, update_tp)
                # new marking has shorter path
                elif new_g < already_visited[new_marking].g:
                    for i in self.open_set:
                        if i.m == new_marking:
                            pre_trans = deepcopy(curr.pre_trans_lst)
                            for j in pre_trans:
                                j.append(incidence_matrix.transitions[t])
                            i, fl = self.update_paths(pre_trans, i)
                            i.g = new_g
                            i.t = t
                            i.p = curr
                            i.propagation_rank = curr.propagation_rank
                            if not i.trust:
                                i = self.derive_or_estimate_heuristic(curr, i, incidence_matrix, cost_vec, t)
                            self.open_set.remove(i)
                            heapq.heappush(self.open_set, i)
                            already_visited[new_marking] = i
                            break
                # new marking has longer or equal path, but the heuristic change from invalid to valid
                else:
                    for i in self.open_set:
                        if i.m == new_marking and not i.trust:
                            i = self.derive_or_estimate_heuristic(curr, i, incidence_matrix, cost_vec, t)
                            # we check all the previous paths and compare with the current paths
                            pre_trans = deepcopy(curr.pre_trans_lst)
                            for j in pre_trans:
                                j.append(incidence_matrix.transitions[t])
                            i, fl = self.update_paths(pre_trans,i)
                            i.propagation_rank = curr.propagation_rank
                            already_visited[new_marking] = i
                            self.open_set.remove(i)
                            heapq.heappush(self.open_set, i)
                            break

    def update_paths(self, pre_trans, to_marking):
        mark = [1 for i in range(len(pre_trans))]
        count = 0
        flag = False
        for j in pre_trans:
            for k in to_marking.pre_trans_lst:
                if (len(j) == len(k) and set(j) == set(k)) or k == j[0:len(k)]:
                    mark[count] = 0
                    continue
            count += 1
        for ele in range(len(mark)):
            if mark[ele] == 1:
                flag = True
                to_marking.pre_trans_lst.append(pre_trans[ele])
        return to_marking, flag

    def get_state(self, state, ini_vec, cost_vec, h):
        state.x, state.trust = self.get_solution_vec(ini_vec, state.pre_trans_lst)
        state.h = self.get_h(h, cost_vec, state.pre_trans_lst)
        state.f = state.g + state.h
        state.propagation_rank = self.propagation_rank
        return state

    def derive_heuristic(self, x, h, t_index, cost):
        y = deepcopy(x)
        y[t_index] -= 1
        return max(0, h - cost), y

    def trust_solution(self, x):
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

    def get_solution_vec(self, ini_vec, pre_trans_lst):
        # get the new solution vec from initial marking
        trust = False
        # check whether there are paths that have valid solution vec
        valid_path = [1 for i in range(len(pre_trans_lst))]
        count = 0
        x = []
        solution_to_add_lst = []
        for trans_lst in pre_trans_lst:
            solution_vec = deepcopy(ini_vec)
            for trans in range(len(trans_lst)):
                solution_vec[trans_lst[trans]] -= 1
                # when the solution vector encounters <0, means no longer trustable
                if solution_vec[trans_lst[trans]] < 0:
                    valid_path[count] = 0
                    break
            solution_to_add_lst.append(solution_vec)
            count += 1
        if 1 in valid_path:
            trust = True
            for i in range(len(valid_path)):
                if valid_path[i] == 1:
                    x.append(solution_to_add_lst[i])
        return x, trust

    def get_h(self, h, cost_vec, pre_tran_lst):
        h_lst = [0]
        for j in pre_tran_lst:
            for i in j:
                h -= cost_vec[i]
            h_lst.append(h)
        return max(h_lst)

    def derive_or_estimate_heuristic(self, from_marking, to_marking, incidence_matrix, cost_vec, t):
        # check all the possible solution vector and store it to to_marking
        x_to_check = deepcopy(from_marking.x)
        # use flag to indicate whether a valid solution vec exists
        valid_flag = False
        to_marking_x_prime = []
        for x_i in x_to_check:
            x_i[incidence_matrix.transitions[t]] -= 1
            # only add valid one to to_marking
            if x_i[incidence_matrix.transitions[t]] >= 0:
                valid_flag = True
                to_marking_x_prime.append(x_i)

        # if from_marking has exact heuristic, we can derive from it
        if from_marking.trust and valid_flag and from_marking.h != "HEURISTICINFINITE":
            to_marking.h = from_marking.h - cost_vec[incidence_matrix.transitions[t]]
            to_marking.f = to_marking.g + to_marking.h
            to_marking.trust = True
            to_marking.x = []
            if from_marking.g <= to_marking.g:
                to_marking.p = from_marking
                to_marking.t = t
            for x_i in to_marking_x_prime:
                to_marking.x.append(x_i)
            if t.label[0] != ">>":
                if self.get_max_events(to_marking) + 1 > self.max_rank:
                    self.max_rank = self.get_max_events(from_marking) + 1

        # if heuristic of from marking is infinite, then we return
        elif from_marking.h == "HEURISTICINFINITE":
            to_marking.h = "HEURISTICINFINITE"
            to_marking.trust = True
            for x_i in to_marking_x_prime:
                to_marking.x.append(x_i)
        else:
            if to_marking.m == self.final_marking:
                to_marking.h = 0
                to_marking.f = to_marking.g
                to_marking.trust = True
                to_marking.t = t
                to_marking.p = from_marking
                for x_i in to_marking_x_prime:
                    to_marking.x.append(x_i)
                if t.label[0] != ">>":
                    if self.get_max_events(to_marking) + 1 > self.max_rank:
                        self.max_rank = self.get_max_events(from_marking) + 1
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
        if marking.t.label[0] != ">>":
            return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())
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
            'visited_states': self.visited,
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


class SearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust, pre_trans_lst, propagation_rank):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        self.pre_trans_lst = pre_trans_lst
        self.propagation_rank = propagation_rank

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
