import timeit

from pm4py.objects.petri_net.utils.align_utils import add_markings
from src.tools import *


class AlignmentWithCounterAstar:
    """
    Alignment computed with split-point-based search algorithm + caching strategy.
    It is based on algorithm proposed in paper [1]. An additional cache set is introduced to store infeasible
    markings explored during search. Also, the marking in close set will be reopened if shorter or equally costly
    sequences are found. Moreover, a counter if used to enforce restart of the search.

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
        trans_num: The number of events in the sync product net
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
        max_counter: The maximum round of not restart.

    References:
    [1] van Dongen, B.F. (2018). Efficiently Computing Alignments.
    """

    def __init__(self, ini, fin, cost_function, incidence_matrix, trace_sync, trace_log, max_counter):
        self.ini = ini
        self.fin = fin
        self.cost_function = cost_function
        self.incidence_matrix = incidence_matrix
        self.trace_sync = trace_sync
        self.trace_log = trace_log
        self.split_open_set_flag = True
        self.trace_len = len(trace_log)
        self.trans_num = len(self.incidence_matrix.transitions)
        self.cost_vec = []
        self.heuristic = 0
        self.open_set = MinHeap()
        self.infeasible_set = {}
        self.visited_state = 0
        self.traversed_arc = 0
        self.simple_lp = 0
        self.complex_lp = 0
        self.restart = 0
        self.max_rank = -2
        self.queue = 0
        self.order = 0
        self.split_lst = []
        self.insertion = 0
        self.update = 0
        self.removal = 0
        self.counter = 0
        self.max_counter = max_counter

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
        self.heuristic += timeit.default_timer() - start_time
        self.simple_lp += 1
        # add initial marking to open set
        ini_state = CacheReopenMarking(h, 0, h, self.ini, None, None,
                                  np.reshape(x, (1, self.trans_num)),
                                  True, self.order,
                                  np.reshape(np.zeros(self.trans_num), (1, self.trans_num)),
                                  len(self.split_lst))
        start_time = timeit.default_timer()
        self.open_set.heap_insert(ini_state)
        self.queue += timeit.default_timer() - start_time
        self.insertion += 1
        self.cost_vec = cost_vec

        while True:
            # while not all states in feasible bucket visited
            while self.open_set.lst:
                self.removal += 1
                start_time = timeit.default_timer()
                new_curr = self.open_set.heap_pop()
                self.queue += timeit.default_timer() - start_time
                # if do not activate infeasible set, then need to compare f-value to the minimum in infeasible.
                curr, flag = self._close_or_update_marking(new_curr, cost_vec, fin_vec)
                if flag == "CLOSEDSUCCESSFUL":
                    closed[curr.m] = curr
                    self.visited_state += 1
                    self._expand_from_current_marking(curr, cost_vec, closed)
                elif flag == "REQUEUED":
                    start_time = timeit.default_timer()
                    self.open_set.heap_insert(curr)
                    self.queue += timeit.default_timer() - start_time
                    self.insertion += 1
                elif flag == "RESTARTNEEDED":
                    self.split_lst = sorted(self.split_lst)
                    start_time = timeit.default_timer()
                    h, x = compute_init_heuristic_with_split(ini_vec, fin_vec, cost_vec, self.split_lst, inc_matrix,
                                                             cons_matrix,
                                                             self.incidence_matrix.transitions,
                                                             self.incidence_matrix.places,
                                                             self.trace_sync, self.trace_log)
                    self.heuristic += timeit.default_timer() - start_time
                    self.counter += 1
                    self.complex_lp += 1
                    if self.counter == self.max_counter:
                        self.counter = 0
                        self.restart += 1
                        closed = {}
                        self.open_set = MinHeap()
                        self.infeasible_set = {}
                        self.split_open_set_flag = False
                        self.order = 0
                        self.max_rank = -2
                        ini_state = CacheReopenMarking(h, 0, h, self.ini, None, None,
                                                  np.reshape(x, (1, self.trans_num)),
                                                  True, self.order,
                                                  np.reshape(np.zeros(self.trans_num), (1, self.trans_num)),
                                                  len(self.split_lst))
                        start_time = timeit.default_timer()
                        self.open_set.heap_insert(ini_state)
                        self.queue += timeit.default_timer() - start_time
                        self.insertion += 1
                    else:
                        cache_set = []
                        feasible_flag = False
                        for each_marking in self.open_set.lst:
                            new_marking = self._update_marking(each_marking, x, h)
                            cache_set.append(new_marking)
                        for each_marking in self.infeasible_set.values():
                            new_marking = self._update_marking(each_marking, x, h)
                            cache_set.append(new_marking)
                        last_max_rank = self.max_rank
                        self.max_rank = -2
                        self.infeasible_set = {}
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
                                    self.insertion += 1
                                else:
                                    self.infeasible_set[new_marking.m] = new_marking
                            self.queue += timeit.default_timer() - start_time
                            self.split_open_set_flag = True
                        else:
                            self.max_rank = last_max_rank
                            for new_marking in cache_set:
                                self.open_set.heap_insert(new_marking)
                                self.insertion += 1
                                self.infeasible_set = {}
                            self.split_open_set_flag = False

                elif flag == "FINALMARKINGFOUND":
                    return self._reconstruct_alignment(curr)
                elif flag == "CLOSEDINFEASIBLE":
                    closed[curr.m] = curr
            if len(self.infeasible_set) > 0:
                # the open set is empty, check whether new split point emerges
                if self.max_rank + 2 < self.trace_len and self.max_rank + 2 not in self.split_lst:
                    self.split_lst.append(self.max_rank + 2)
                    self.split_lst = sorted(self.split_lst)
                    start_time = timeit.default_timer()
                    h, x = compute_init_heuristic_with_split(ini_vec, fin_vec, cost_vec, self.split_lst, inc_matrix,
                                                             cons_matrix,
                                                             self.incidence_matrix.transitions,
                                                             self.incidence_matrix.places,
                                                             self.trace_sync, self.trace_log)
                    self.heuristic += timeit.default_timer() - start_time
                    self.complex_lp += 1
                    self.counter += 1
                    if self.counter == self.max_counter:
                        self.counter = 0
                        self.restart += 1
                        closed = {}
                        self.open_set = MinHeap()
                        self.infeasible_set = {}
                        self.split_open_set_flag = False
                        self.order = 0
                        self.max_rank = -2
                        ini_state = CacheReopenMarking(h, 0, h, self.ini, None, None,
                                                  np.reshape(x, (1, self.trans_num)),
                                                  True, self.order,
                                                  np.reshape(np.zeros(self.trans_num), (1, self.trans_num)),
                                                  len(self.split_lst))
                        start_time = timeit.default_timer()
                        self.open_set.heap_insert(ini_state)
                        self.queue += timeit.default_timer() - start_time
                        self.insertion += 1
                    else:
                        # add all states in open set and infeasible set to cache set
                        cache_set = []
                        feasible_flag = False
                        last_max_rank = self.max_rank
                        self.max_rank = -2
                        for each_marking in self.open_set.lst:
                            new_marking = self._update_marking(each_marking, x, h)
                            cache_set.append(new_marking)
                        for each_marking in self.infeasible_set.values():
                            new_marking = self._update_marking(each_marking, x, h)
                            cache_set.append(new_marking)
                        self.infeasible_set = {}
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
                                    self.insertion += 1
                                else:
                                    self.infeasible_set[new_marking.m] = new_marking
                            self.queue += timeit.default_timer() - start_time
                            self.split_open_set_flag = True
                        # if some markings in cache set become feasible, then put feasible markings to open set,
                        # infeasible markings to infeasible set
                        else:
                            self.max_rank = last_max_rank

                            for new_marking in cache_set:
                                self.open_set.heap_insert(new_marking)
                                self.insertion += 1
                                self.infeasible_set = {}
                            self.split_open_set_flag = False
                else:
                    # the trusted open set is empty, have to turn to infeasible set
                    # put all of them into open set and set the use infeasible flag to true
                    self.insertion += len(self.infeasible_set)
                    self.queued += len(self.infeasible_set)
                    start_time = timeit.default_timer()
                    for v in self.infeasible_set.values():
                        self.open_set.heap_insert(v)
                    self.queue += timeit.default_timer() - start_time
                    self.infeasible_set = {}
                    self.split_open_set_flag = False

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
            self.heuristic += timeit.default_timer() - start_time
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
                marking.x = np.reshape(x, (1, self.trans_num))
                marking.trust = True
                marking.heuristic_priority += 1
                # need to requeue the marking
                return marking, "REQUEUED"
            else:
                self.simple_lp += 1
                # continue with this marking
                marking.f = marking.g + h
                marking.h = h
                marking.heuristic_priority += 1
                marking.x = np.reshape(x, (1, self.trans_num))
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
            t_idx = self.incidence_matrix.transitions[t]
            # compute the new g score of the subsequent marking reached if t would be fired
            new_g = curr.g + cost
            new_m = add_markings(curr.m, t.add_marking)
            self.traversed_arc += 1
            # subsequent marking is fresh, compute the f score of this path and add it to open set
            if new_m in closed:
                marking_to_explore = closed[new_m]
                # reach this marking with shorter path, abandon all longer paths
                if marking_to_explore.g > new_g:
                    t_idx = self.incidence_matrix.transitions[t]
                    del closed[new_m]
                    marking_to_explore.p, marking_to_explore.t, marking_to_explore.g = curr, t, new_g
                    # update previous transition list and abandon path before
                    new_parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                    marking_to_explore.parikh_vec_lst = new_parikh_vec_lst
                    new_h, new_x, trustable = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                    marking_to_explore.h = new_h
                    marking_to_explore.x = new_x
                    marking_to_explore.trust = trustable
                    marking_to_explore.f = marking_to_explore.g + new_h
                    start_time = timeit.default_timer()
                    self.queue += timeit.default_timer() - start_time
                    self.update += 1
                    if trustable or not self.split_open_set_flag:
                        if trustable:
                            marking_to_explore.heuristic_priority = curr.heuristic_priority
                            self.max_rank = check_max_event(marking_to_explore, self.max_rank, t)
                        start_time = timeit.default_timer()
                        self.open_set.heap_insert(marking_to_explore)
                        self.queue += timeit.default_timer() - start_time
                        self.insertion += 1
                    else:
                        self.infeasible_set[new_m] = marking_to_explore
                elif marking_to_explore.g == new_g:
                    # update previous paths
                    temp_parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                    new_parikh_vec_lst, update_flag = update_parikh_vec_lst(temp_parikh_vec_lst,
                                                                                   marking_to_explore.parikh_vec_lst)
                    # if new paths found
                    if update_flag:
                        del closed[new_m]
                        marking_to_explore.parikh_vec_lst = new_parikh_vec_lst
                        # compute all possible solution vector
                        new_h, new_x, trustable = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                        # if the h is still feasible, keep it in open set, otherwise put into infeasible set
                        if trustable or not self.split_open_set_flag:
                            if trustable:
                                if curr.heuristic_priority > marking_to_explore.heuristic_priority:
                                    marking_to_explore.heuristic_priority = curr.heuristic_priority
                                    marking_to_explore.x = new_x
                                else:
                                    marking_to_explore.x = concatenate_two_sol(marking_to_explore.x, new_x)
                            marking_to_explore.h = min(marking_to_explore.h, new_h)
                            marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                            start_time = timeit.default_timer()
                            self.open_set.heap_insert(marking_to_explore)
                            self.queue += timeit.default_timer() - start_time
                            self.update += 1
                        else:
                            self.infeasible_set[new_m] = marking_to_explore

            else:
                # check if m is in feasible set
                m_in_open_flag = self.open_set.heap_find(new_m)
                # check if m is in infeasible set
                m_in_infeasible_set = self.split_open_set_flag and new_m in self.infeasible_set
                # split open set into two: set containing feasible markings and set containing infeasible markings
                if self.split_open_set_flag:
                    # if this marking is already in open set (feasible set)
                    if m_in_open_flag:
                        start_time = timeit.default_timer()
                        marking_to_explore = self.open_set.heap_get(new_m)
                        self.queue += timeit.default_timer() - start_time

                        # reach this marking with shorter path, abandon all longer paths
                        if new_g < marking_to_explore.g:
                            marking_to_explore.p, marking_to_explore.t, marking_to_explore.g = curr, t, new_g
                            # update previous transition list
                            marking_to_explore.parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                            # get heuristic and solution vector
                            new_h, new_x, trustable = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                            marking_to_explore.h = new_h
                            marking_to_explore.x = new_x
                            marking_to_explore.trust = trustable
                            marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                            # if the h is still feasible, keep it in open set, otherwise put into infeasible set
                            if trustable:
                                start_time = timeit.default_timer()
                                self.open_set.heap_update(marking_to_explore)
                                self.queue += timeit.default_timer() - start_time
                                self.update += 1
                            else:
                                self.open_set.heap_remove(marking_to_explore)
                                self.infeasible_set[new_m] = marking_to_explore
                        # if found a path with equal length, need to propagate the new path
                        elif new_g == marking_to_explore.g:
                            # update previous transition list
                            temp_parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                            new_parikh_vec_lst, update_flag = update_parikh_vec_lst(temp_parikh_vec_lst,
                                                                                           marking_to_explore.parikh_vec_lst)
                            # if new paths found
                            if update_flag:
                                marking_to_explore.parikh_vec_lst = new_parikh_vec_lst
                                # compute all possible solution vector
                                new_h, new_x, trustable = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                                # if the h is still feasible, keep it in open set, otherwise put into infeasible set
                                if trustable:
                                    marking_to_explore.h = min(marking_to_explore.h, new_h)
                                    marking_to_explore.x = concatenate_two_sol(marking_to_explore.x, new_x)
                                    marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                                    start_time = timeit.default_timer()
                                    self.open_set.heap_update(marking_to_explore)
                                    self.queue += timeit.default_timer() - start_time
                                    self.update += 1
                    # subsequent marking is not in open set but in infeasible set
                    elif m_in_infeasible_set:
                        marking_to_explore = self.infeasible_set[new_m]
                        # if paths are updated
                        if new_g < marking_to_explore.g:
                            marking_to_explore.p, marking_to_explore.t, marking_to_explore.g = curr, t, new_g
                            # update previous transition list
                            marking_to_explore.parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                            new_h, new_x, trustable = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                            marking_to_explore.h = new_h
                            marking_to_explore.x = new_x
                            marking_to_explore.trust = trustable
                            marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                            if trustable:
                                # remove from infeasible set
                                del self.infeasible_set[new_m]
                                start_time = timeit.default_timer()
                                self.open_set.heap_insert(marking_to_explore)
                                self.queue += timeit.default_timer() - start_time
                                self.update += 1
                                self.insertion += 1
                            else:
                                self.infeasible_set[new_m] = marking_to_explore
                        elif new_g == marking_to_explore.g:
                            # update previous transition list
                            temp_parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                            new_parikh_vec_lst, update_flag = update_parikh_vec_lst(temp_parikh_vec_lst,
                                                                                           marking_to_explore.parikh_vec_lst)
                            # reach the marking with a different paths but has same g-value
                            if update_flag:
                                # compute all possible solution vector
                                marking_to_explore.parikh_vec_lst = new_parikh_vec_lst
                                new_h, new_x, trustable = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                                # if the h is still feasible, keep it in open set, otherwise put into infeasible set
                                if trustable:
                                    marking_to_explore.h = new_h
                                    marking_to_explore.x = new_x
                                    marking_to_explore.trust = True
                                    marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                                    # remove from infeasible set
                                    del self.infeasible_set[new_m]
                                    start_time = timeit.default_timer()
                                    self.open_set.heap_insert(marking_to_explore)
                                    self.queue += timeit.default_timer() - start_time
                                    self.insertion += 1
                                else:
                                    marking_to_explore.h = min(marking_to_explore.h, new_h)
                                    marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                                    self.infeasible_set[new_m] = marking_to_explore
                    # explore marking for the first time, namely, neither in open set or infeasible set
                    else:
                        # update previous transition list
                        new_parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                        t_idx = self.incidence_matrix.transitions[t]
                        new_h, new_x, trustable = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                        # if the h is still feasible, keep it in open set, otherwise put into infeasible set
                        if trustable:
                            tp = CacheReopenMarking(new_g + new_h, new_g, new_h, new_m, curr, t, new_x, trustable,
                                               self.order, new_parikh_vec_lst, curr.heuristic_priority)

                            start_time = timeit.default_timer()
                            self.open_set.heap_insert(tp)
                            self.queue += timeit.default_timer() - start_time
                            self.insertion += 1
                            self.max_rank = check_max_event(tp, self.max_rank, t)
                        else:
                            tp = CacheReopenMarking(new_g + new_h, new_g, new_h, new_m, curr, t, [], trustable,
                                               self.order, new_parikh_vec_lst, curr.heuristic_priority)
                            self.infeasible_set[new_m] = tp
                # put all markings into open set, regardless of feasibleity of markings
                else:
                    if m_in_open_flag:
                        start_time = timeit.default_timer()
                        marking_to_explore = self.open_set.heap_get(new_m)
                        self.queue += timeit.default_timer() - start_time
                        # if shorter path found, update

                        if new_g < marking_to_explore.g:
                            marking_to_explore.p, marking_to_explore.t, marking_to_explore.g = curr, t, new_g
                            # update previous transition list and abandon path before
                            new_parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                            marking_to_explore.parikh_vec_lst = new_parikh_vec_lst
                            new_h, new_x, trustable = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                            marking_to_explore.h = new_h
                            marking_to_explore.x = new_x
                            marking_to_explore.trust = trustable
                            marking_to_explore.f = marking_to_explore.g + new_h
                            start_time = timeit.default_timer()
                            self.open_set.heap_update(marking_to_explore)
                            self.queue += timeit.default_timer() - start_time
                            self.update += 1
                        # subsequent marking has equal path, but the heuristic change from infeasible to feasible
                        elif new_g == marking_to_explore.g:
                            temp_parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                            new_parikh_vec_lst, update_flag = update_parikh_vec_lst(temp_parikh_vec_lst,
                                                                                           marking_to_explore.parikh_vec_lst)
                            # if new paths are found
                            if update_flag:
                                marking_to_explore.parikh_vec_lst = new_parikh_vec_lst
                                new_h, new_x, trust_flag = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                                # if the path is equally long, but the heuristic change from infeasible to feasible
                                if not marking_to_explore.trust and trust_flag:
                                    marking_to_explore.h = new_h
                                    marking_to_explore.f = marking_to_explore.h + marking_to_explore.g
                                    marking_to_explore.x = new_x
                                    marking_to_explore.trust_flag = True
                                    start_time = timeit.default_timer()
                                    self.open_set.heap_update(marking_to_explore)
                                    self.queue += timeit.default_timer() - start_time
                                    self.update += 1
                                elif marking_to_explore.trust and trust_flag:
                                    marking_to_explore.h = min(marking_to_explore.h, new_h)
                                    marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                                    marking_to_explore.x = concatenate_two_sol(marking_to_explore.x, new_x)
                                    start_time = timeit.default_timer()
                                    self.open_set.heap_update(marking_to_explore)
                                    self.queue += timeit.default_timer() - start_time
                                    self.update += 1
                    # the marking explored is not in open set
                    else:
                        self.order += 1
                        new_h, new_x, trustable = derive_multi_heuristic(cost_vec, curr.x, t_idx, curr.h)
                        new_parikh_vec_lst = get_parikh_vec_lst(curr.parikh_vec_lst, t_idx)
                        tp = CacheReopenMarking(new_g + new_h, new_g, new_h, new_m, curr, t, new_x, trustable,
                                           self.order, new_parikh_vec_lst, curr.heuristic_priority)
                        if trustable:
                            self.max_rank = check_max_event(tp, self.max_rank, t)
                        start_time = timeit.default_timer()
                        self.open_set.heap_insert(tp)
                        self.queue += timeit.default_timer() - start_time
                        self.insertion += 1
    
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
            "heuristic": self.heuristic,
            "queue": self.queue,
            "num_insert": self.insertion,
            "num_removal": self.removal,
            "num_update": self.update,
            'sum': self.update + self.insertion + self.removal,
            'states': self.visited_state,
            'arcs': self.traversed_arc,
            'split_num': len(self.split_lst),
            'trace_length': self.trace_len,
            'alignment_length': len(alignment),
            'alignment': alignment,
        }

    def _update_marking(self, marking, new_sol, h):

        # get the new solution vec from initial marking
        trust_flag = False
        # use h_to_add_lst to store all possible h
        # use feasible_solution_lst to store all possible solution vec
        feasible_solution_lst = []
        feasible_h_lst = []
        infeasible_h_lst = []

        for each_lst in marking.parikh_vec_lst:
            new_solution_vec, new_h, trustable = derive_heuristic_from_ini(new_sol, each_lst, self.cost_vec, h)
            if trustable:
                feasible_solution_lst.append(new_solution_vec)
                feasible_h_lst.append(max(new_h, 0))
                trust_flag = True
            else:
                infeasible_h_lst.append(max(new_h, 0))

        # if feasible h exists
        if trust_flag:
            marking.h = min(feasible_h_lst)
            marking.x = np.array(feasible_solution_lst)
            marking.heuristic_priority = len(self.split_lst)
        # if none of h is feasible
        else:
            marking.h = min(infeasible_h_lst)
            marking.x = []
        marking.f = marking.g + marking.h
        marking.trust = trust_flag
        return marking




