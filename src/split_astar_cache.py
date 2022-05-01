import timeit

from pm4py.objects.petri_net.utils.align_utils import add_markings
from src.tools import *


class AlignmentWithCacheAstar:
    """
    Alignment computed with split-point-based search algorithm + caching strategy.
    It is based on algorithm proposed in paper [1].

    Attributes:
        ini: The initial marking
        fin: The final marking
        cost_function: the cost function used
        incidence_matrix: The incidence matrix of synchronous product net
        trace_sync: The list of index for sync move
        trace_log: The list of index for log move
        split_lst: The list of split points
        split_open_set_flag: If true, use an additional cache set, otherwise not
        trace_len: The number of event in the trace
        open_set: The open set used for search
        cache_set: The cache set to store infeasible markings
        trace_sync: The list of index for sync move
        trace_log: The list of index for log move
        visited_state: The number of markings visited during search
        traversed_arcs: The number of arcs traversed during search
        simple_lp: The number of regular lp solved
        complex_lp: The number of complex lp with split point list solved
        restart: The number of restart during search
        heuristic_time: The time spent on computing h-value
        queue_time: The time spent on open set related operation
        num_insert: The num of marking insertion into open set
        num_removal: The num of marking removal from open set
        num_update: The num of marking update from open set
        order: The order of marking explored
        max_rank: The max index of event explained so far

    References:
    [1] van Dongen, B.F. (2018). Efficiently Computing Alignments.
    """

    def __init__(self, ini, fin, cost_function, incidence_matrix, trace_sync, trace_log):
        self.ini = ini
        self.fin = fin
        self.cost_function = cost_function
        self.incidence_matrix = incidence_matrix
        self.trace_sync = trace_sync
        self.trace_log = trace_log
        self.split_lst = []
        self.split_open_set_flag = True
        self.trace_len = len(trace_log)
        self.open_set = MinHeap()
        self.cache_set = {}
        self.visited_state = 0
        self.traversed_arc = 0
        self.simple_lp = 0
        self.complex_lp = 0
        self.restart = 0
        self.heuristic_time = 0
        self.queue_time = 0
        self.order = 0
        self.num_insert = 0
        self.num_update = 0
        self.num_removal = 0
        self.max_rank = -2

    def search(self):
        """
        Find a shortest path in the sync product net as alignment result.

        Returns
        -------
        alignment: dict
            The alignment result and other metrics recorded during search
        """

        ini_vec, fin_vec, cost_vec = vectorize_initial_final_cost(self.incidence_matrix, self.ini,
                                                                  self.fin, self.cost_function)
        closed = {}
        inc_matrix = self.incidence_matrix.a_matrix
        cons_matrix = self.incidence_matrix.b_matrix
        start_time = timeit.default_timer()
        h, x = compute_init_heuristic_without_split(fin_vec - ini_vec, inc_matrix, cost_vec)
        self.heuristic_time += timeit.default_timer() - start_time
        self.simple_lp += 1
        trans_num = len(self.incidence_matrix.transitions)
        ini_state = CacheMarking(h, 0, h, self.ini, None, None, x, True, self.order, np.zeros(trans_num))
        start_time = timeit.default_timer()
        self.open_set.heap_insert(ini_state)
        self.queue_time += timeit.default_timer() - start_time
        self.num_insert += 1

        while True:
            # while not all states in feasible bucket visited
            while self.open_set.lst:
                self.num_removal += 1
                start_time = timeit.default_timer()
                new_curr = self.open_set.heap_pop()
                self.queue_time += timeit.default_timer() - start_time
                marking_diff = fin_vec - self.incidence_matrix.encode_marking(new_curr.m)
                curr, flag = self._close_or_update_marking(new_curr, cost_vec, fin_vec)
                if flag == "CLOSEDSUCCESSFUL":
                    closed[curr.m] = curr
                    self.visited_state += 1
                    self._expand_from_current_marking(curr, cost_vec, closed)
                elif flag == "REQUEUED":
                    start_time = timeit.default_timer()
                    self.open_set.heap_insert(curr)
                    self.queue_time += timeit.default_timer() - start_time
                    self.num_insert += 1
                elif flag == "RESTARTNEEDED":
                    self.split_lst = sorted(self.split_lst)
                    start_time = timeit.default_timer()
                    h, x = compute_init_heuristic_with_split(ini_vec, fin_vec, cost_vec, self.split_lst,
                                                             inc_matrix, cons_matrix,
                                                             self.incidence_matrix.transitions,
                                                             self.incidence_matrix.places,
                                                             self.trace_sync, self.trace_log)
                    self.heuristic_time += timeit.default_timer() - start_time
                    self.complex_lp += 1
                    cache_set = []
                    feasible_flag = False
                    last_max_rank = self.max_rank
                    self.max_rank = -2
                    for each_marking in self.open_set.lst:
                        new_marking = self._update_marking(each_marking, x, h)
                        cache_set.append(new_marking)
                    for each_marking in self.cache_set.values():
                        new_marking = self._update_marking(each_marking, x, h)
                        cache_set.append(new_marking)
                    self.cache_set = {}
                    self.open_set = MinHeap()
                    # update each marking and add to cache set
                    if curr.m not in closed:
                        new_curr_marking = self._update_marking(curr, x, h)
                        cache_set.append(new_curr_marking)
                    # put marking in cache set into open set
                    for new_marking in cache_set:
                        if new_marking.trust:
                            feasible_flag = True
                            self.max_rank = max(get_max_events(new_marking), self.max_rank)
                    # if no markings in cache set become feasible, then fall back to normal A*
                    if feasible_flag:
                        start_time = timeit.default_timer()
                        for new_marking in cache_set:
                            if new_marking.trust:
                                self.open_set.heap_insert(new_marking)
                                self.num_insert += 1
                            else:
                                self.cache_set[new_marking.m] = new_marking
                        self.queue_time += timeit.default_timer() - start_time
                        self.split_open_set_flag = True
                    else:
                        self.max_rank = last_max_rank
                        for new_marking in cache_set:
                            self.open_set.heap_insert(new_marking)
                            self.num_insert += 1
                            self.cache_set = {}
                        self.split_open_set_flag = False
                elif flag == "FINALMARKINGFOUND":
                    return self._reconstruct_alignment(curr)
                elif flag == "CLOSEDINFEASIBLE":
                    closed[curr.m] = curr

            if len(self.cache_set) > 0:
                # the open set is empty, then need to check whether new split point emerges
                if self.max_rank + 2 < self.trace_len and self.max_rank + 2 not in self.split_lst:
                    self.split_lst.append(self.max_rank + 2)
                    self.split_lst = sorted(self.split_lst)
                    start_time = timeit.default_timer()
                    h, x = compute_init_heuristic_with_split(ini_vec, fin_vec, cost_vec, self.split_lst,
                                                             inc_matrix, cons_matrix,
                                                             self.incidence_matrix.transitions,
                                                             self.incidence_matrix.places,
                                                             self.trace_sync, self.trace_log)
                    self.heuristic_time += timeit.default_timer() - start_time
                    self.complex_lp += 1
                    last_max_rank = self.max_rank
                    self.max_rank = -2

                    # add all states in open set and infeasible set to cache set
                    cache_set = []
                    feasible_flag = False

                    for each_marking in self.open_set.lst:
                        new_marking = self._update_marking(each_marking, x, h)
                        cache_set.append(new_marking)
                    for each_marking in self.cache_set.values():
                        new_marking = self._update_marking(each_marking, x, h)
                        cache_set.append(new_marking)
                    self.cache_set = {}
                    self.open_set = MinHeap()
                    # update each marking and add to cache set
                    if curr.m not in closed:
                        new_curr_marking = self._update_marking(curr, x, h)
                        cache_set.append(new_curr_marking)
                    # put marking in cache set into open set
                    for new_marking in cache_set:
                        if new_marking.trust:
                            feasible_flag = True
                            self.max_rank = max(get_max_events(new_marking), self.max_rank)
                    # if no markings in cache set become feasible, then put all of them into open set
                    if feasible_flag:
                        start_time = timeit.default_timer()
                        for new_marking in cache_set:
                            if new_marking.trust:
                                self.open_set.heap_insert(new_marking)
                                self.num_insert += 1
                            else:
                                self.cache_set[new_marking.m] = new_marking
                        self.queue_time += timeit.default_timer() - start_time
                        self.split_open_set_flag = True
                    # if some markings in cache set become feasible, then put feasible markings to open set,
                    # infeasible markings to infeasible set
                    else:
                        self.max_rank = last_max_rank
                        for new_marking in cache_set:
                            self.open_set.heap_insert(new_marking)
                            self.num_insert += 1
                            self.cache_set = {}
                        self.split_open_set_flag = False
                        # restart by reset open set and closed set
                else:
                    # the trusted open set is empty, have to turn to infeasible set
                    # put all of them into open set and set the use infeasible flag to true
                    self.num_insert += len(self.cache_set)
                    start_time = timeit.default_timer()
                    for v in self.cache_set.values():
                        self.open_set.heap_insert(v)
                    self.queue_time += timeit.default_timer() - start_time
                    # clear the infeasible set and reset min f
                    self.cache_set = {}
                    self.split_open_set_flag = False

    # compute solutions for the state
    def _update_marking(self, marking, new_sol, h):
        # get the new solution vec from initial marking
        new_solution_vec = new_sol.copy()
        # get the new solution vector
        new_solution_vec = new_solution_vec - marking.parikh_vec_lst
        marking.h = max(h - marking.g, 0)
        marking.f = marking.g + marking.h

        # the heuristic is not feasible
        if np.any(new_solution_vec < 0):
            marking.x = []
            marking.trust = False
        else:
            marking.x = new_solution_vec
            marking.trust = True
        return marking


    def _close_or_update_marking(self, marking, cost_vec, fin_vec):
        """
        Put marking in closed set if marking is feasible, otherwise compute an exact h-value for it.
        """

        if marking.m == self.fin and marking.h == 0:
            return marking, "FINALMARKINGFOUND"
        if not marking.trust:
            # compute the exact heuristics
            start_time = timeit.default_timer()
            marking_diff = fin_vec - self.incidence_matrix.encode_marking(marking.m)
            h, x, trustable, self.split_lst, self.max_rank = \
                compute_new_heuristic(marking, self.split_lst, marking_diff, self.ini, self.incidence_matrix.a_matrix,
                                      cost_vec, self.max_rank, self.trace_len)
            self.heuristic_time += timeit.default_timer() - start_time
            if h == -1:
                # need to restart
                return marking, "RESTARTNEEDED"
            # heuristic is not computable, from which final marking is unreachable
            elif trustable == "Infeasible":
                self.simple_lp += 1
                return marking, "CLOSEDINFEASIBLE"
            # if the heuristic is higher push the head of the queue down, set the score to exact score
            elif h > marking.h:
                self.simple_lp += 1
                marking.f = marking.g + h
                marking.h = h
                marking.x = x
                marking.trust = True
                # need to requeue the marking
                return marking, "REQUEUED"
            else:
                self.simple_lp += 1
                # continue with this marking
                marking.f = marking.g + h
                marking.h = h
                marking.x = x
                marking.trust = trustable
        return marking, "CLOSEDSUCCESSFUL"

    def _expand_from_current_marking(self, curr, cost_vec, closed):
        """
        Expand all subsequent markings from current marking.
        """

        enabled_trans1 = {}
        for p in curr.m:
            for t in p.ass_trans:
                if t.sub_marking <= curr.m:
                    enabled_trans1[t] = self.incidence_matrix.transitions[t]
                    enabled_trans1 = dict(sorted(enabled_trans1.items(), key=lambda item: item[1]))
        enabled_trans = enabled_trans1.keys()
        trans_to_visit_with_cost = [(t, self.cost_function[t]) for t in enabled_trans if not (
                t is not None and is_log_move(t, '>>') and is_model_move(t, '>>'))]
        for t, cost in trans_to_visit_with_cost:
            # compute the new g score of the subsequent marking reached if t would be fired
            new_g = curr.g + cost
            new_m = add_markings(curr.m, t.add_marking)
            self.traversed_arc += 1
            # subsequent marking is fresh, compute the f score of this path and add it to open set
            if new_m not in closed:
                # check if m is in feasible set
                m_in_open_flag = self.open_set.heap_find(new_m)
                # check if m is in infeasible set
                m_in_cache_set = self.split_open_set_flag and new_m in self.cache_set
                # split open set into two: set containing feasible markings and set containing infeasible markings
                if self.split_open_set_flag:
                    # if this marking is already in open set (feasible set)
                    if m_in_open_flag:
                        start_time = timeit.default_timer()
                        marking_to_explore = self.open_set.heap_get(new_m)
                        self.queue_time += timeit.default_timer() - start_time
                        # reach this marking with shorter path, abandon all longer paths
                        if new_g < marking_to_explore.g:
                            marking_to_explore.p, marking_to_explore.t, marking_to_explore.g = curr, t, new_g
                            # update previous transition list
                            marking_to_explore.parikh_vec_lst = get_parikh_vec(curr.parikh_vec_lst,
                                                                               self.incidence_matrix.transitions[t])
                            # get heuristic and solution vector
                            new_h, new_x, trustable = derive_heuristic(cost_vec, curr.x,
                                                                       self.incidence_matrix.transitions[t], curr.h)
                            marking_to_explore.h = new_h
                            marking_to_explore.x = new_x
                            marking_to_explore.trust = trustable
                            marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                            # if the h is still feasible, keep it in open set, otherwise put into infeasible set
                            if trustable:
                                start_time = timeit.default_timer()
                                self.open_set.heap_update(marking_to_explore)
                                self.queue_time += timeit.default_timer() - start_time
                                self.num_update += 1
                            else:
                                self.open_set.heap_remove(marking_to_explore)
                                self.cache_set[new_m] = marking_to_explore
                    # subsequent marking is not in open set but in infeasible set
                    elif m_in_cache_set:
                        marking_to_explore = self.cache_set[new_m]
                        # if paths are updated
                        if new_g < marking_to_explore.g:
                            marking_to_explore.p, marking_to_explore.t, marking_to_explore.g = curr, t, new_g
                            # update previous transition list
                            marking_to_explore.parikh_vec_lst = get_parikh_vec(curr.parikh_vec_lst,
                                                                               self.incidence_matrix.transitions[t])
                            new_h, new_x, trustable = derive_heuristic(cost_vec, curr.x,
                                                                       self.incidence_matrix.transitions[t], curr.h)
                            marking_to_explore.h = new_h
                            marking_to_explore.x = new_x
                            marking_to_explore.trust = trustable
                            marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                            if trustable:
                                # remove from infeasible set
                                del self.cache_set[new_m]
                                start_time = timeit.default_timer()
                                self.open_set.heap_insert(marking_to_explore)
                                self.queue_time += timeit.default_timer() - start_time
                                self.num_update += 1
                            else:
                                self.cache_set[new_m] = marking_to_explore
                        elif new_g == marking_to_explore.g:
                            # update previous transition list
                            new_pre_trans = get_parikh_vec(curr.parikh_vec_lst, self.incidence_matrix.transitions[t])
                            # compute all possible solution vector
                            new_h, new_x, trustable = derive_heuristic(cost_vec, curr.x,
                                                                       self.incidence_matrix.transitions[t], curr.h)
                            # if the h is still feasible, keep it in open set, otherwise put into infeasible set
                            if trustable:
                                marking_to_explore.h = new_h
                                marking_to_explore.x = new_x
                                marking_to_explore.trust = True
                                marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                                marking_to_explore.parikh_vec_lst = new_pre_trans
                                del self.cache_set[new_m]
                                start_time = timeit.default_timer()
                                self.open_set.heap_insert(marking_to_explore)
                                self.queue_time += timeit.default_timer() - start_time
                                self.num_update += 1
                            else:
                                marking_to_explore.h = min(marking_to_explore.h, new_h)
                                marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                                self.cache_set[new_m] = marking_to_explore
                    # explore marking for the first time, namely, neither in open set or infeasible set
                    else:
                        # update previous transition list
                        new_pre_trans = get_parikh_vec(curr.parikh_vec_lst, self.incidence_matrix.transitions[t])
                        new_h, new_x, trustable = derive_heuristic(cost_vec, curr.x,
                                                                   self.incidence_matrix.transitions[t], curr.h)
                        tp = CacheMarking(new_g + new_h, new_g, new_h, new_m, curr, t, new_x, trustable,
                                          self.order, new_pre_trans)
                        # if the h is still feasible, keep it in open set, otherwise put into infeasible set
                        if trustable:
                            start_time = timeit.default_timer()
                            self.open_set.heap_insert(tp)
                            self.queue_time += timeit.default_timer() - start_time
                            self.num_update += 1
                            self.max_rank = check_max_event(tp, self.max_rank, t)
                        else:
                            self.cache_set[new_m] = tp

                # put all markings into open set, regardless of feasibleity of markings
                else:
                    if m_in_open_flag:
                        start_time = timeit.default_timer()
                        marking_to_explore = self.open_set.heap_get(new_m)
                        self.queue_time += timeit.default_timer() - start_time
                        # if shorter path found, update
                        if new_g < marking_to_explore.g:
                            marking_to_explore.p, marking_to_explore.t, marking_to_explore.g = curr, t, new_g

                            # update previous transition list and abandon path before
                            marking_to_explore.parikh_vec_lst = get_parikh_vec(curr.parikh_vec_lst,
                                                                               self.incidence_matrix.transitions[t])
                            new_h, new_x, trustable = derive_heuristic(cost_vec, curr.x,
                                                                       self.incidence_matrix.transitions[t], curr.h)
                            marking_to_explore.h = new_h
                            marking_to_explore.x = new_x
                            marking_to_explore.trust = trustable
                            marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                            start_time = timeit.default_timer()
                            self.open_set.heap_update(marking_to_explore)
                            self.queue_time += timeit.default_timer() - start_time
                            self.num_update += 1
                        # subsequent marking has equal path, but the heuristic change from infeasible to feasible
                        elif new_g == marking_to_explore.g:
                            new_pre_trans = get_parikh_vec(curr.parikh_vec_lst, self.incidence_matrix.transitions[t])
                            new_h, new_x, trust_flag = derive_heuristic(cost_vec, curr.x,
                                                                        self.incidence_matrix.transitions[t], curr.h)
                            # if the path is equally long, but the heuristic change from infeasible to feasible
                            if not marking_to_explore.trust and trust_flag:
                                marking_to_explore.h = new_h
                                marking_to_explore.f = marking_to_explore.h + marking_to_explore.g
                                marking_to_explore.x = new_x
                                marking_to_explore.parikh_vec_lst = new_pre_trans
                                marking_to_explore.trust_flag = True
                                start_time = timeit.default_timer()
                                self.open_set.heap_update(marking_to_explore)
                                self.queue_time += timeit.default_timer() - start_time
                                self.num_update += 1
                    # the marking explored is not in open set
                    else:
                        self.order += 1
                        new_h, new_x, trustable = derive_heuristic(cost_vec, curr.x,
                                                                   self.incidence_matrix.transitions[t], curr.h)
                        new_pre_trans = get_parikh_vec(curr.parikh_vec_lst, self.incidence_matrix.transitions[t])

                        tp = CacheMarking(new_g + new_h, new_g, new_h, new_m, curr, t, new_x, trustable,
                                          self.order, new_pre_trans)
                        if trustable:
                            self.max_rank = check_max_event(tp, self.max_rank, t)
                        start_time = timeit.default_timer()
                        self.open_set.heap_insert(tp)
                        self.queue_time += timeit.default_timer() - start_time
                        self.num_insert += 1

    def _reconstruct_alignment(self, state, ret_tuple_as_trans_desc=False):
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
            'cost': state.f,
            'simple_lp': self.simple_lp,
            "complex_lp": self.complex_lp,
            'restart': self.restart,
            "heuristic": self.heuristic_time,
            "queue": self.queue_time,
            "num_insert": self.num_insert,
            "num_removal": self.num_removal,
            "num_update": self.num_update,
            'sum': self.num_update + self.num_insert + self.num_removal,
            'states': self.visited_state,
            'arcs': self.traversed_arc,
            'split_num': len(self.split_lst),
            'trace_length': self.trace_len,
            'alignment_length': len(alignment),
            'alignment': alignment,
        }
