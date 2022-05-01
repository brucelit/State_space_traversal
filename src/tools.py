"""
Define necessary class and function for alignment computation.
class IncidenceMatrix, Parameters, SynchronousProduct are introduced with slight modification from PM4Py.

References:
    https://github.com/pm4py
"""


import sys
import re
import numpy as np
import gurobipy as gp

from gurobipy import GRB
from enum import Enum
from copy import copy
from pm4py.objects.petri import align_utils as utils
from pm4py.objects.petri.utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.objects.petri_net.obj import PetriNet, Marking as SyncMarking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.objects.petri_net import properties


def compute_init_heuristic_with_split(ini_vec,
                                      fin_vec,
                                      cost,
                                      split_lst,
                                      incidence_matrix,
                                      consumption_matrix,
                                      t_index,
                                      p_index,
                                      trace_lst_sync,
                                      trace_lst_log):
    """
    When the search gets stuck, get the new heuristic for initial marking

    Parameters
    ----------
    ini_vec: numpy array
      The vector of the initial marking
    fin_vec: numpy array
      The vector of the final marking
    cost: numpy array
      The cost function
    incidence_matrix: numpy 2-D array
      The incidence matrix of synchronous product net
    consumption_matrix: numpy 2-D array
      The consumption matrix of synchronous product net
    t_index: dict
      The dictionary of transition and corresponding index for sync net
    p_index: dict
      The dictionary of place and corresponding index for sync net
    trace_lst_sync: list
      The list of transition index for sync move
    trace_lst_log: list
      The list of transition index for log move
    Returns
    -------
    heuristic:
      The heuristic value for the marking
    solution vector:
      The solution vector for the marking
    """

    # Define lp problem
    m = gp.Model()
    m.Params.LogToConsole = 0
    # Create two 2-D arrays of integer variables X and Y, 0 to k+1
    k = len(split_lst)
    # constraint 3, note that the variable type can be non-integer
    x = m.addMVar((k + 1, len(t_index)), vtype=GRB.INTEGER, lb=0)
    y = m.addMVar((k + 1, len(t_index)), vtype=GRB.INTEGER, lb=0)
    # Set objective
    m.setObjective(sum(np.array(cost) @ x[i, :] + np.array(cost) @ y[i, :] for i in range(k + 1)), GRB.MINIMIZE)
    # Add constraint 1
    cons_one = np.array([0 for i in range(len(p_index))])
    for i in range(k + 1):
        sum_x = incidence_matrix @ x[i, :]
        sum_y = incidence_matrix @ y[i, :]
        cons_one += sum_x + sum_y
    m.addConstr(cons_one == fin_vec - ini_vec)
    # Add constraint 2
    cons_two_temp = incidence_matrix @ x[0, :]
    for a in range(1, k + 1):
        for b in range(1, a):
            if b == a - 1:
                cons_two_temp += incidence_matrix @ x[b, :] + incidence_matrix @ y[b, :]
        ct2 = cons_two_temp + consumption_matrix @ y[a, :]
        m.addConstr(ct2 + ini_vec >= 0)
    # Add constraints 4, 5 and 6:
    y_col = 1
    m.addConstr(y[0, :].sum() == 0)
    for i in split_lst:
        if len(trace_lst_sync[i]) > 0 and trace_lst_log[i] is not None:
            y_index = 0
            for j in trace_lst_sync[i]:
                y_index += y[y_col, j]
            y_index += y[y_col, trace_lst_log[i]]
            m.addConstr(y_index == 1)
        else:
            k_1 = trace_lst_log[i]
            m.addConstr(y[y_col, k_1] == 1)
        m.addConstr(y[y_col, :].sum() == 1)
        y_col += 1
    # optimize model
    m.optimize()
    return m.objVal, list(np.array(x.X).sum(axis=0) + np.array(y.X).sum(axis=0))


def derive_multi_heuristic(cost_vec, solution_lst, t_idx, h):
    """
    Given list of solution vector of previous marking, derive solution vec list for subsequent marking.

    Parameters
    ----------
    cost_vec : dict
               the unit cost function
    solution_lst : list
                   the list of solution vector
    t_idx : int
            the index of transition
    h : int
        h-value
    Returns
    -------
    h : int
        h-value
    solution_lst : list
                   list of solution vectors
    h_feasibility : Boolean
                    True if feasible solution exists, otherwise False
    """

    new_solution_lst = []
    infeasible_h_lst = []
    feasible_h_lst = []
    solution_vec_lst = solution_lst.copy()
    # for every solution vector in solution list, check whether it is feasible
    for x_prime in solution_vec_lst:
        if x_prime[t_idx] >= 1:
            x_prime[t_idx] -= 1
            new_solution_lst.append(x_prime)
            feasible_h_lst.append(max(h - cost_vec[t_idx], 0))
        else:
            infeasible_h_lst.append(max(h - cost_vec[t_idx], 0))
    if len(new_solution_lst) > 0:
        return min(feasible_h_lst), np.array(new_solution_lst), True
    else:
        return min(infeasible_h_lst), [], False


def compute_init_heuristic_without_split(marking_diff, incidence_matrix, cost_vec):
    """
    When the search starts, get the heuristic for initial marking without any split

    Parameters
    ----------
    marking_diff : numpy array
                   The vector difference of final marking and initial marking
    incidence_matrix : numpy 2-D array
                       The incidence matrix of synchronous product net
    cost_vec: numpy array
              The cost function

    Returns
    -------
    heuristic:
      The heuristic value for the marking
    solution vector:
      The solution vector for the marking
    """


    m = gp.Model()
    m.Params.LogToConsole = 0
    x = m.addMVar((1, len(cost_vec)), vtype=GRB.INTEGER, lb=0)
    z = np.array(incidence_matrix) @ x[0, :]
    m.addConstr(z == marking_diff)
    m.setObjective(sum(cost_vec @ x[i, :] for i in range(1)), GRB.MINIMIZE)
    m.optimize()
    return m.objVal, list(np.array(x.X.sum(axis=0)))


def compute_new_heuristic(marking, split_lst, marking_diff, ini, incidence_matrix,
                          cost_vec, max_rank, trace_len):
    """
    Given an infeasible marking, update the new split list if new split point is found.
    If no new split point is found, compute an exact heuristic for this infeasible marking.

    Parameters
    ----------
    marking :Marking
        The infeasible marking to handle
    split_lst: list
        List of split points so far
    marking_diff: numpy array
        The difference between two marking vectors
    ini: initial marking
        The initial marking
    incidence_matrix: numpy 2-D array
        The incidence matrix of synchronous product net
    cost_vec: numpy array
        The cost function
    max_rank: int
        The maximum event index explored so far
    trace_len: int
        The length of trace
    Returns
    -------
    heuristic:
        The heuristic value for the marking
    solution vector:
        The solution vector for the marking
    trusted:
        The feasibility of heuristic
    split_list:
        The updated split list
    max_rank:
        The index of the max event explained
    """

    insert_position = 1
    rank = max_rank + 1
    if marking.m != ini:
        if rank + 1 not in split_lst:
            insert_position = -1

    # if:
    # 1. marking equals initial marking
    # 2. the index of max event is already in split list
    # 3. the index of max event is greater than trace length
    if marking.m == ini or insert_position > 0 or rank + 1 >= trace_len:
        m = gp.Model()
        m.Params.LogToConsole = 0
        x = m.addMVar((1, len(cost_vec)), vtype=GRB.INTEGER, lb=0)
        z = np.array(incidence_matrix) @ x[0, :]
        m.addConstr(z == marking_diff)
        m.setObjective(sum(cost_vec @ x[i, :] for i in range(1)), GRB.MINIMIZE)
        m.optimize()
        r = get_max_events(marking)
        if r > max_rank:
            max_rank = r
        if m.status == 2:
            return m.objVal, list(np.array(x.X.sum(axis=0))), True, split_lst, max_rank
        else:
            return "HEURISTICINFINITE", [], "Infeasible", split_lst, max_rank
    else:
        split_lst.append(rank + 1)
        return -1, [0 for i in range(len(cost_vec))], False, split_lst, max_rank


def derive_heuristic(cost_vec, x, t_idx, h):
    """
    derive new h-value.
    """
    if x[t_idx] >= 1:
        trustable = True
        new_x = x.copy()
        new_x[t_idx] -= 1
    else:
        new_x = []
        trustable = False
    return max(0, h - cost_vec[t_idx]), new_x, trustable


def derive_heuristic_from_ini(new_sol, each_lst, cost_vec, h):
    new_sol_vec = new_sol.copy()
    new_sol_vec2 = new_sol_vec - each_lst
    h -= each_lst @ cost_vec
    if np.min(new_sol_vec2) < 0:
        return [], h, False
    else:
        return new_sol_vec2, h, True


def check_max_event(marking, max_rank, t):
    """
    update max event explained.
    """
    if t.label[0] != ">>":
        temp_max = get_max_events(marking)
        if temp_max > max_rank:
            max_rank = temp_max
    return max_rank


def get_parikh_vec(parikh_vec, t_idx):
    """
    Get the Parikh vector from previous marking.
    """
    new_parikh_vec = parikh_vec.copy()
    new_parikh_vec[t_idx] += 1
    return new_parikh_vec


def get_sequence_len(marking):
    """
    Get the length of the sequence leading to the marking.
    """
    if marking.p is None:
        return 0
    else:
        return 1 + get_sequence_len(marking.p)


def get_max_events(marking):
    """
    Get the index of the max event explained for the marking.
    If no events explained so far, return -2.
    """
    if marking.t is None:
        return -2
    if marking.t.label[0] != ">>":
        return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())
    return get_max_events(marking.p)


def get_parikh_vec_lst(parikh_vec_lst, t_idx):
    """
    Get the list of parikh vector.
    """
    new_parikh_vec_lst = parikh_vec_lst.copy()
    for each_lst in new_parikh_vec_lst:
        each_lst[t_idx] += 1
    return new_parikh_vec_lst


def update_parikh_vec_lst(parikh_vec1, parikh_vec2):
    """
    With new Parikh vector(s) found, update the list of Parikh vector. Iterate previous known
    list of Parikh vectors, and check whether new Parikh vector is already included.

    Parameters
    ----------
    parikh_vec1 : list
                  previous known list of parikh vector
    parikh_vec2 : list
                  new list of parikh vector

    Returns
    -------
    parikh_vec2 : list
                  updated list of parikh vector
    update_flag : boolean
                  if the list of parikh vector is updated, return true, otherwise false.
    """
    update_flag = False
    for each_new_path in parikh_vec1:
        in_flag = False
        for each_old_path in parikh_vec2:
            if (each_new_path >= each_old_path).all():
                in_flag = True
                break
        if not in_flag:
            update_flag = True
            parikh_vec2 = np.vstack((parikh_vec2, each_new_path))
    return parikh_vec2, update_flag


def is_model_move(t, skip):
    return t.label[0] == skip and t.label[1] != skip


def is_log_move(t, skip):
    return t.label[0] != skip and t.label[1] == skip


def construct_incidence_matrix(net):
    return IncidenceMatrix(net)


def copy_into(source_net, target_net, upper, skip):
    t_map = {}
    p_map = {}
    lst_t = []
    lst_p = []
    for p in source_net.places:
        lst_p.append(p)
    for t in source_net.transitions:
        lst_t.append(t)
    lst_t.sort(key=lambda k: k.name)
    lst_p.sort(key=lambda k: k.name)

    for t in lst_t:
        name = (t.name, skip) if upper else (skip, t.name)
        label = (t.label, skip) if upper else (skip, t.label)
        t_map[t] = PetriNet.Transition(name, label)
        if properties.TRACE_NET_TRANS_INDEX in t.properties:
            # 16/02/2021: copy the index property from the transition of the trace net
            t_map[t].properties[properties.TRACE_NET_TRANS_INDEX] = t.properties[properties.TRACE_NET_TRANS_INDEX]
        target_net.transitions.add(t_map[t])

    for p in lst_p:
        name = (p.name, skip) if upper else (skip, p.name)
        p_map[p] = PetriNet.Place(name)
        if properties.TRACE_NET_PLACE_INDEX in p.properties:
            # 16/02/2021: copy the index property from the place of the trace net
            p_map[p].properties[properties.TRACE_NET_PLACE_INDEX] = p.properties[properties.TRACE_NET_PLACE_INDEX]
        target_net.places.add(p_map[p])

    for t in lst_t:
        for a in t.in_arcs:
            add_arc_from_to(p_map[a.source], t_map[t], target_net)
        for a in t.out_arcs:
            add_arc_from_to(t_map[t], p_map[a.target], target_net)
    return t_map, p_map


def concatenate_two_sol(x, new_x):
    for each_new_x in new_x:
        in_flag = False
        for each_x in x:
            if (each_new_x == each_x).all():
                in_flag = True
                break
        if not in_flag:
            x = np.vstack((x, new_x))
    return x


def vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function):
    ini_vec = incidence_matrix.encode_marking(ini)
    fin_vec = incidence_matrix.encode_marking(fin)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[incidence_matrix.transitions[t]] = cost_function[t]
    return np.array(ini_vec), np.array(fin_vec), np.array(cost_vec)


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


class SynchronousProduct:
    def __init__(self, trace, petri_net, initial_marking, final_marking):
        self.trace = trace
        self.petri_net = petri_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.incidence_matrix = []
        self.trace_log = []
        self.trace_sync = []
        self.cost_vec = []
        self.cost_function = []

    def construct_cost_aware(self, pn1, im1, fm1, pn2, im2, fm2, skip, pn1_costs, pn2_costs, sync_costs):
        """
        Constructs the synchronous product net of two given Petri nets.
        :param pn1: Petri net 1
        :param im1: Initial marking of Petri net 1
        :param fm1: Final marking of Petri net 1
        :param pn2: Petri net 2
        :param im2: Initial marking of Petri net 2
        :param fm2: Final marking of Petri net 2
        :param skip: Symbol to be used as skip
        :param pn1_costs: dictionary mapping transitions of pn1 to corresponding costs
        :param pn2_costs: dictionary mapping transitions of pn2 to corresponding costs
        :param pn1_costs: dictionary mapping pairs of transitions in pn1 and pn2 to costs
        :param sync_costs: Costs of sync moves

        Returns
        -------
        :return: Synchronous product net and associated marking labels are of the form (a,>>)
        """
        sync_net = PetriNet('synchronous_product_net')
        t1_map, p1_map = copy_into(pn1, sync_net, True, skip)
        t2_map, p2_map = copy_into(pn2, sync_net, False, skip)
        costs = dict()
        lst_t_pn1 = []
        lst_t_pn2 = []
        for t in pn1.transitions:
            lst_t_pn1.append(t)
        for t in pn2.transitions:
            lst_t_pn2.append(t)
        lst_t_pn1.sort(key=lambda k: k.name)
        lst_t_pn2.sort(key=lambda k: k.name)

        for t1 in lst_t_pn1:
            costs[t1_map[t1]] = pn1_costs[t1]
        for t2 in lst_t_pn2:
            costs[t2_map[t2]] = pn2_costs[t2]
        for t1 in lst_t_pn1:
            for t2 in lst_t_pn2:
                if t1.label == t2.label:
                    sync = PetriNet.Transition((t1.name, t2.name), (t1.label, t2.label))
                    sync_net.transitions.add(sync)
                    costs[sync] = sync_costs[(t1, t2)]
                    # copy the properties of the transitions inside the transition of the sync net
                    for p1 in t1.properties:
                        sync.properties[p1] = t1.properties[p1]
                    for p2 in t2.properties:
                        sync.properties[p2] = t2.properties[p2]
                    for a in t1.in_arcs:
                        add_arc_from_to(p1_map[a.source], sync, sync_net)
                    for a in t2.in_arcs:
                        add_arc_from_to(p2_map[a.source], sync, sync_net)
                    for a in t1.out_arcs:
                        add_arc_from_to(sync, p1_map[a.target], sync_net)
                    for a in t2.out_arcs:
                        add_arc_from_to(sync, p2_map[a.target], sync_net)

        sync_im = SyncMarking()
        sync_fm = SyncMarking()
        for p in im1:
            sync_im[p1_map[p]] = im1[p]
        for p in im2:
            sync_im[p2_map[p]] = im2[p]
        for p in fm1:
            sync_fm[p1_map[p]] = fm1[p]
        for p in fm2:
            sync_fm[p2_map[p]] = fm2[p]

        # update 06/02/2021: to distinguish the sync nets that are output of this method, put a property in the sync net
        sync_net.properties[properties.IS_SYNC_NET] = True
        return sync_net, sync_im, sync_fm, costs

    def construct_sync_product(self, trace, petri_net, initial_marking, final_marking, parameters=None):
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
                map(lambda e: 1, trace))
            parameters[Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

        if model_cost_function is None:
            # reset variables value
            model_cost_function = dict()
            sync_cost_function = dict()

            # apply unit cost function to assign cost for each move
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

        sync_prod, sync_initial_marking, sync_final_marking, cost_function = self.construct_cost_aware(
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
        decorate_transitions_prepostset(sync_prod)
        decorate_places_preset_trans(sync_prod)
        incidence_matrix = construct_incidence_matrix(sync_prod)
        trace_sync = [[] for i in range(0, len(trace_lst))]
        trace_log = [None for i in range(0, len(trace_lst))]
        t_index = incidence_matrix.transitions
        for t in sync_prod.transitions:
            for i in range(len(trace_lst)):
                if trace_lst[i].name == t.name[0] and t.label[1] == ">>":
                    trace_log[i] = t_index[t]
                if trace_lst[i].name == t.name[0] and t.label[1] != ">>":
                    trace_sync[i].append(t_index[t])
        return sync_initial_marking, sync_final_marking, cost_function, incidence_matrix, trace_sync, trace_log


class MinHeap:

    def __init__(self):
        self.lst = []
        self.idx = {}

    def _swap(self, idx1, idx2):
        self.lst[idx1], self.lst[idx2] = self.lst[idx2], self.lst[idx1]
        self.idx[self.lst[idx1].m], self.idx[self.lst[idx2].m] = self.idx[self.lst[idx2].m], self.idx[self.lst[idx1].m]

    def heap_insert(self, marking):
        self.lst.append(marking)
        self.idx[marking.m] = len(self.lst) - 1
        self._heap_heapify_up(len(self.lst) - 1)

    def heap_pop(self):
        if len(self.lst) > 1:
            marking_to_pop = self.lst[0]
            del self.idx[marking_to_pop.m]
            # update list and index
            self.lst[0] = self.lst[-1]
            self.idx[self.lst[0].m] = 0
            # remove the last element
            self.lst.pop()
            self._heap_heapify_down(0)
            return marking_to_pop
        else:
            marking_to_pop = self.lst[0]
            self.heap_clear()
            return marking_to_pop

    def heap_remove(self, marking):
        idx_to_remove = self.idx[marking.m]
        if len(self.lst) > 1:
            if idx_to_remove == len(self.lst) - 1:
                marking_to_remove = self.lst[idx_to_remove]
                del self.idx[marking_to_remove.m]
                self.lst.pop()
            else:
                marking_to_remove = self.lst[idx_to_remove]
                del self.idx[marking_to_remove.m]
                # update list and index
                self.lst[idx_to_remove] = self.lst[-1]
                self.idx[self.lst[idx_to_remove].m] = idx_to_remove
                # remove the last element
                self.lst.pop()
                self._heap_heapify_down(idx_to_remove)
        else:
            self.heap_clear()

    def heap_find(self, m):
        return True if m in self.idx else False

    def heap_get(self, m):
        return self.lst[self.idx[m]]

    def heap_update(self, marking):
        idx_to_update = self.idx[marking.m]
        self.lst[idx_to_update] = marking
        self._heap_heapify_down(idx_to_update)
        self._heap_heapify_up(idx_to_update)

    def heap_clear(self):
        self.lst.clear()
        self.idx.clear()

    def _heap_heapify_down(self, idx):
        while self._has_left_child(idx):
            smaller_index = idx * 2 + 1
            if idx * 2 + 2 < len(self.lst) and self.lst[idx * 2 + 2] < self.lst[idx * 2 + 1]:
                smaller_index = idx * 2 + 2
            if self.lst[idx] < self.lst[smaller_index]:
                break
            else:
                self._swap(smaller_index, idx)
            idx = smaller_index

    def _heap_heapify_up(self, idx):
        index = idx
        while index != 0 and self.lst[index] < self.lst[int((index - 1) // 2)]:
            parent_index = int((index - 1) // 2)
            self._swap(parent_index, index)
            index = parent_index

    def _has_left_child(self, idx):
        if idx * 2 + 1 < len(self.lst):
            return True
        else:
            return False


class NormalMarking:
    """
    The marking used for split-point-based algorithm.

    Attributes:
        f: f-value
        g: g-value
        h: h-value
        m: a multiset of places
        p: previous marking
        t: previous transition
        x: solution vector
        trust: the feasibility of h-value, true if h is feasible
        order: the of the first visit this marking
    """

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
        # first order criterion is f score
        if self.f != other.f:
            return self.f < other.f
        # second order criterion on exactness of heuristic
        if self.trust != other.trust:
            return self.trust
        # third criterion: prefer marking with the larger event explained
        max_event1 = get_max_events(self)
        max_event2 = get_max_events(other)
        if max_event1 != max_event2:
            return max_event1 > max_event2
        # fourth order criterion is g score
        if self.g > other.g:
            return True
        elif self.g < other.g:
            return False
        # prefer longer path
        path1 = get_sequence_len(self)
        path2 = get_sequence_len(other)
        if path1 != path2:
            return path1 > path2
        # prefer the marking reached later
        return self.order > other.order


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


class CacheMarking:
    """
    The marking used for split-point-based algorithm with caching strategy.

    Attributes:
        f: f-value
        g: g-value
        h: h-value
        m: a multiset of places
        p: previous marking
        t: previous transition
        x: list of feasible solution vector
        trust: the feasibility of h-value, true if h is feasible
        order: the of the first visit this marking
        parikh_vec_lst: list of parikh vector
    """
    def __init__(self, f, g, h, m, p, t, x, trust, order, parikh_vec_lst):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        self.order = order
        self.parikh_vec_lst = parikh_vec_lst

    def __lt__(self, other):
        # first order criterion is f score, prefer marking with smaller f-value.
        if self.f != other.f:
            return self.f < other.f
        # second order criterion is feasibility of h, prefer marking with feasible h.
        if self.trust != other.trust:
            return self.trust
        # third criterion: prefer marking with the larger event explained.
        max_event1 = get_max_events(self)
        max_event2 = get_max_events(other)
        if max_event1 != max_event2:
            return max_event1 > max_event2
        # fourth order criterion is g score, prefer marking with larger g-value.
        if self.g > other.g:
            return True
        elif self.g < other.g:
            return False
        # prefer marking with longer previous path
        path1 = get_sequence_len(self)
        path2 = get_sequence_len(other)
        if path1 != path2:
            return path1 > path2
        return self.order > other.order

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


class CacheReopenMarking:
    """
    The marking used for split-point-based algorithm with caching strategy and reopen method.

    Attributes:
        f: f-value
        g: g-value
        h: h-value
        m: a multiset of places
        p: previous marking
        t: previous transition
        x: list of feasible solution vector
        trust: the feasibility of h-value, true if h is feasible
        order: the of the first visit this marking
        parikh_vec_lst: list of parikh vector
        heuristic_priority: the state of heuristic. If heuristic if updated later, also update this value
    """
    def __init__(self, f, g, h, m, p, t, x, trust, order, parikh_vec_lst, heuristic_priority):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        self.order = order
        self.parikh_vec_lst = parikh_vec_lst
        self.heuristic_priority = heuristic_priority

    def __lt__(self, other):
        # first order criterion is f score
        if self.f != other.f:
            return self.f < other.f
        # second order criterion on exactness of heuristic
        if self.trust != other.trust:
            return self.trust
        # third criterion: prefer marking with the larger event explained
        max_event1 = get_max_events(self)
        max_event2 = get_max_events(other)
        if max_event1 != max_event2:
            return max_event1 > max_event2
        # fourth order criterion is g score
        if self.g > other.g:
            return True
        elif self.g < other.g:
            return False
        # prefer longer path
        path1 = get_sequence_len(self)
        path2 = get_sequence_len(other)
        if path1 != path2:
            return path1 > path2
        # prefer the marking reached later
        return self.order > other.order

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
        transitions_lst = sorted([x for x in net.transitions], key=lambda x: (str(x.name[0][1]), id(x)))

        transitions = []
        sync_trans = [0 for i in range(len(transitions_lst))]
        log_trans = [0 for i in range(len(transitions_lst))]
        for i in transitions_lst:
            if i.label[0] == '>>':
                transitions.append(i)
        for i in transitions_lst:
            if i.label[1] == '>>':
                key1 = int(re.search("(\d+)(?!.*\d)", i.name[0]).group())
                log_trans[key1] = i
        for i in log_trans:
            if i == 0:
                continue
            transitions.append(i)
        for i in transitions_lst:
            if i.label[0] != '>>' and i.label[1] != ">>":
                key1 = int(re.search("(\d+)(?!.*\d)", i.name[0]).group())
                sync_trans[key1] = i

        for i in sync_trans:
            if i == 0:
                continue
            transitions.append(i)

        for p in places:
            p_index[p] = len(p_index)
        for t in transitions:
            t_index[t] = len(t_index)
        p_index_sort = sorted(p_index.items(), key=lambda kv: kv[0].name, reverse=True)
        new_p_index = dict()
        for i in range(len(p_index_sort)):
            new_p_index[p_index_sort[i][0]] = i
        a_matrix = np.array([[0 for i in range(len(t_index))] for j in range(len(new_p_index))])
        b_matrix = np.array([[0 for i in range(len(t_index))] for j in range(len(new_p_index))])
        for p in net.places:
            for a in p.in_arcs:
                a_matrix[new_p_index[p]][t_index[a.source]] += 1
            for a in p.out_arcs:
                a_matrix[new_p_index[p]][t_index[a.target]] -= 1
                b_matrix[new_p_index[p]][t_index[a.target]] -= 1
        return a_matrix, b_matrix, new_p_index, t_index

    a_matrix = property(__get_a_matrix)
    b_matrix = property(__get_b_matrix)
    places = property(__get_place_indices)
    transitions = property(__get_transition_indices)
