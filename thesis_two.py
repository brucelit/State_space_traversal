import copy
import heapq

from pm4py.objects.petri import align_utils as utils

from astar_implementation import heuristic, util
from astar_implementation import utilities as utilities
from astar_implementation import visualization

ret_tuple_as_trans_desc = False


def astar_with_split(sync_net, sync_im, sync_fm, aux_dict, split_lst):
    """
    ------------
    # Line 1-10
    ------------
     """
    ini_vec, fin_vec, cost_vec = utilities.vectorize_initial_final_cost(sync_im, sync_fm,
                                                                        aux_dict['p_index'], aux_dict['t_index'],
                                                                        aux_dict['cost_function'])
    closed_set = set()  # init close set
    heuristic_set = set()  # init estimated heuristic set
    closed_set2 = set()

    #  initial state
    ini_state = init_state(sync_im, ini_vec, fin_vec, cost_vec, split_lst,
                           aux_dict['incidence_matrix'], aux_dict['consumption_matrix'], aux_dict['t_index'])
    open_set = []
    heapq.heapify(open_set)
    heapq.heappush(open_set, ini_state)

    # init the number of states explained
    new_split_point = None
    max_num = 0

    # matrice for measurement
    queued = 0
    traversed = 0
    valid_state_lst = set()
    invalid_state_lst = set()

    # dict_g来表示他是否被访问过, key是marking_tuple
    dict_g = {ini_state.marking_tuple: 0}

    # use state_path to represent all the possible path leading to current state
    state_path = {ini_state.marking_tuple: [[]]}

    print("before checking:", )
    for i in aux_dict['state_to_check']:
        print(i.marking, "not trust:", i.not_trust)
    print("after checking:")
    changed_state, valid_path = check_state(aux_dict['state_to_check'], ini_state.parikh_vector)
    for i in changed_state:
        print(i.marking)
    for j in valid_path:
        print(j)

    '''
    ------------------
    # Line 10-30  
    ------------------
    '''
    while len(open_set) > 0:
        aux_dict['visited'] += 1
        aux_dict['order'] += 1
        print("\norder:", aux_dict['order'])
        print("open set:")
        heapq.heapify(open_set)
        for i in open_set:
            print(i.marking, "f:", i.f, "not trust:", i.not_trust, "g:", i.g, "h:", i.h)
        heapq.heapify(open_set)
        #  favouring markings for which the exact heuristic is known.
        curr = heapq.heappop(open_set)

        print("\ncurrent marking", curr.marking)
        # print("current not trust", curr.not_trust)
        # print("current f,g,h", curr.f, curr.g, curr.h)
        # print("current parikh", curr.parikh_vector)
        if curr.marking_tuple in heuristic_set:
            print("Its in heuristic set")
        # current_marking = copy.deepcopy(curr.marking)
        # tranform places in current state to list in form p1,p2,p3…
        curr_state_lst = [marking_to_list(curr.marking, aux_dict['place_map'])]

        lst2 = marking_to_list(curr.marking, aux_dict['place_map'])
        if curr.not_trust == 0 and lst2 not in valid_state_lst:
            valid_state_lst.add(lst2)
            if lst2 in invalid_state_lst:
                invalid_state_lst.remove(lst2)
        if curr.not_trust == 1 and lst2 not in valid_state_lst and lst2 not in invalid_state_lst:
            invalid_state_lst.add(lst2)
        # visualize and save change
        gviz = visualization.viz_state_change(aux_dict['ts'], curr_state_lst, valid_state_lst, invalid_state_lst,
                                              aux_dict['visited'], split_lst, open_set)
        # ts_visualizer.save(gviz, os.path.join("E:/Thesis/img/acegcd", "step" + str(order) + ".png"))

        curr_vec = utilities.encode_marking(curr.marking, aux_dict['p_index'])
        if curr_vec == fin_vec:
            result = print_result(curr, aux_dict['visited'], queued, traversed, split_lst)
            return result

        if curr.marking_tuple in heuristic_set:
            print("find in heuristic set")
            # print("solution vector",curr.parikh_vector)
            if new_split_point not in split_lst:
                print("order", aux_dict['order'], "new split point", new_split_point, "max num:", max_num)
                split_lst.append(new_split_point)
                heapq.heappush(open_set, curr)
                state_to_check = []
                for i in open_set:
                    if i.not_trust == 1:
                        state_to_check.append(i)
                        print(i.marking, i.pre_trans_lst)
                print("test part:")
                for i in state_to_check:
                    for j in i.pre_trans_lst:
                        print("trans list index", j)
                return astar_with_split(sync_net, sync_im, sync_fm, aux_dict, split_lst)

            new_heuristic, new_parikh_vector = heuristic.compute_exact_heuristic(curr_vec, fin_vec,
                                                                                 aux_dict['incidence_matrix'], cost_vec)
            heuristic_set.remove(curr.marking_tuple)
            old_h = curr.h
            curr.not_trust = 0
            curr.parikh_vector = new_parikh_vector
            curr.h = new_heuristic
            if lst2 not in valid_state_lst:
                valid_state_lst.add(lst2)
                if lst2 in invalid_state_lst:
                    invalid_state_lst.remove(lst2)
            if new_heuristic > old_h:
                curr.f = curr.g + new_heuristic
                # print("new f", curr.f)
                # print("new h", curr.h)
                # print("new not trust", curr.not_trust)
                heapq.heappush(open_set, curr)
                # requeue the state after recalculating
                continue
        # add marking to closed set
        closed_set.add(curr.marking_tuple)
        closed_set2.add(curr.marking)
        # keep track of the maximum number of events explained
        print("closed set", closed_set2)
        new_max_num = max_events_explained(curr, aux_dict['sync_trans'])
        if new_max_num > max_num:
            max_num = new_max_num
            new_split_point = curr.last_sync
            print("new split:", new_split_point, aux_dict['t_index'][new_split_point])
        '''
        -------------
        # Line 31-end  
        -------------
        '''
        # compute enabled transitions and apply model move restriction
        enabled_trans = compute_enabled_transition(curr)

        for t in enabled_trans:
            new_marking = utils.add_markings(curr.marking, t.add_marking)
            new_tuple = tuple(utilities.encode_marking(new_marking, aux_dict['p_index']))
            # reach a marking not yet visited
            if new_tuple not in dict_g:
                dict_g[new_tuple] = 10000
            if new_tuple not in closed_set:
                traversed += 1
                # create new state
                not_trust, new_heuristic, new_parikh_vector, a, t, new_pre_trans_lst, last_sync = \
                    compute_new_state_test(curr, cost_vec, t, aux_dict['t_index'])
                if new_tuple not in state_path:
                    state_path[new_tuple] = new_pre_trans_lst
                else:
                    for i in new_pre_trans_lst:
                        state_path[new_tuple].append(i)
                print("new marking:", new_marking)
                print("path for new state", state_path[new_tuple])

                if a <= dict_g[new_tuple]:
                    g = a
                    if not_trust == 1:
                        f = g + max(0, curr.h - cost_vec[aux_dict['t_index'][t]])
                        h = new_heuristic
                    else:
                        f = g + new_heuristic
                        h = new_heuristic
                    new_state = util.State(not_trust, f, g, h, new_marking, new_tuple, curr, t, new_pre_trans_lst,
                                           last_sync, new_parikh_vector)
                    lst2 = marking_to_list(new_marking, aux_dict['place_map'])
                    if not_trust == 0:
                        if lst2 in invalid_state_lst:
                            print("yes, remove!!!")
                            invalid_state_lst.remove(lst2)
                            heuristic_set.remove(new_state.marking_tuple)
                        valid_state_lst.add(lst2)
                    else:
                        heuristic_set.add(new_state.marking_tuple)
                        invalid_state_lst.add(lst2)
                        if lst2 in valid_state_lst:
                            heuristic_set.remove(new_state.marking_tuple)
                            invalid_state_lst.remove(lst2)
                    dict_g[new_tuple] = new_state.g
                # whether add this state to open set
                flag = 1
                for i in open_set:
                    if i.marking_tuple == new_tuple:
                        if f < i.f and not not_trust:
                            print("success remove", i.marking)
                            open_set.remove(i)
                            queued -= 1
                            heapq.heapify(open_set)
                        elif f == i.f and i.not_trust and not not_trust:
                            print("success remove", i.marking)
                            open_set.remove(i)
                            queued -= 1
                            heapq.heapify(open_set)
                        else:
                            flag = 0
                if flag == 1:
                    heapq.heappush(open_set, new_state)
                    queued += 1

        gviz = visualization.viz_state_change(aux_dict['ts'], curr_state_lst, valid_state_lst, invalid_state_lst,
                                              aux_dict['visited'],
                                              split_lst, open_set)
        print("\nNumber of states visited: " + str(aux_dict['visited']) + "\nsplit list:" + str(
            split_lst[1:]) + "\nNumber of states in open set: " + str(len(open_set)) +
              "\nvalid state: ", valid_state_lst, "\ninvalid state", invalid_state_lst)
        # ts_visualizer.save(gviz, os.path.join("E:/Thesis/img/acegcd", "step" + str(order) + ".png"))


def check_state(state_to_check, ini_parikh_vector):
    changed_state = []
    valid_path = []
    for state in state_to_check:
        for path in state.pre_trans_lst:
            temp_not_trust = 0
            print(path)
            for trans in path:
                if ini_parikh_vector[trans] < 1:
                    temp_not_trust = 1
            if temp_not_trust == 0:
                state.not_trust = 0
                changed_state.append(state)
                valid_path.append(path)
    return changed_state, valid_path


def max_events_explained(curr, sync_trans):
    max_count = 0
    count = 0
    # get the split point
    for lst in curr.pre_trans_lst:
        for i in lst:
            if i in sync_trans:
                count += 1
        if count > max_count:
            max_count = count
    return max_count


def init_state(sync_im, ini_vec, fin_vec, cost_vec, split_lst, incidence_matrix, consumption_matrix, t_index):
    ini_h, ini_parikh_vector = heuristic.compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix,
                                                               consumption_matrix, split_lst, t_index)
    ini_tuple = tuple(ini_vec)
    ini_f = ini_h
    pre_trans_lst = [[]]
    print("ini_h:", ini_h)
    print("solution vector:", ini_parikh_vector)
    # not_trust equal to 0 means the solution vector is known.
    not_trust = 0
    return util.State(not_trust, ini_f, 0, ini_h, sync_im, ini_tuple, None, None, pre_trans_lst, None,
                      ini_parikh_vector)


def compute_new_state_test(curr, cost_vec, t, t_index):
    # for i in curr.pre_trans_lst:
    new_pre_trans_lst = copy.deepcopy(curr.pre_trans_lst)
    for i in new_pre_trans_lst:
        index_to_add = t_index[t]
        i.append(index_to_add)
    new_h_score, new_parikh_vector, not_trust = heuristic.compute_estimate_heuristic(curr.h, curr.parikh_vector,
                                                                                     t_index[t], cost_vec)
    # print("current marking", curr.marking)
    # print("solution vector for curr:", curr.parikh_vector)
    # print("current g", curr.g)
    # print("new marking", new_marking)
    # print("new solution vector:", new_parikh_vector)
    if t is not None and (t.label[0] == t.label[1]):
        last_sync = t
    else:
        last_sync = curr.last_sync
    # compute cost so far
    a = curr.g + cost_vec[t_index[t]]
    print("previous trans for new state", new_pre_trans_lst)
    return not_trust, new_h_score, new_parikh_vector, a, t, new_pre_trans_lst, last_sync


def marking_to_list(marking, place_map):
    element_valid_state = []
    for p in sorted(marking, key=lambda k: k.name[1]):
        element_valid_state.append(place_map[p])
    lst2 = '[%s]' % ', '.join(map(str, element_valid_state))
    return lst2


def compute_enabled_transition(state):
    possible_enabling_transitions = set()
    for p in state.marking:
        for t in p.ass_trans:
            possible_enabling_transitions.add(t)
    enabled_trans = [t for t in possible_enabling_transitions if t.sub_marking <= state.marking]
    violated_trans = []
    for t in enabled_trans:
        if state.pre_transition is None:
            break
        if state.pre_transition.label[1] == ">>" and t.label[0] == ">>":
            violated_trans.append(t)
    for t in violated_trans:
        enabled_trans.remove(t)
    return sorted(enabled_trans, key=lambda k: k.label)


def print_result(state, visited, queued, traversed, split_lst):
    result = utilities.reconstruct_alignment(state, visited, queued, traversed, False)
    print("Optimal alignment:", result["alignment"], "\nCost of optimal alignment:",
          result["cost"], "\nNumber of states visited:", result["visited_states"],
          "\nNumber of split: " + str(len(split_lst) - 1) + "\nTransition in split set: " + \
          str(split_lst[1:]) + "\nF-score for final state: " + \
          str(state.f) + "\nNumber of states visited: " + str(visited) + \
          "\nNumber of states queued: " + str(queued) + \
          "\nNumber of edges traversed: " + str(traversed))
    return result
