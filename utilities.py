import numpy as np
from scipy.optimize import linprog
from pm4py.objects import petri
from pm4py.objects.petri import utils as petri_utils
from pm4py.objects.petri.petrinet import PetriNet, Marking
import pulp

def reconstruct_alignment(state, visited, queued, traversed, ret_tuple_as_trans_desc=False):
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
    return {'alignment': alignment, 'cost': state.g, 'visited_states': visited, 'queued_states': queued,
            'traversed_arcs': traversed}


def compute_estimated_heuristic(marking_source_vec, marking_destination_vec, incidence_matrix, cost_vec):
    marking_diff = np.array(marking_destination_vec) - np.array(marking_source_vec)
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)
    trans_num = len(cost_vec)
    place_num = len(marking_destination_vec)
    var = np.array([pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(trans_num)])
    prob += pulp.lpDot(cost_vec, var)
    ct1 = np.dot(incidence_matrix, var)
    for i in range(place_num):
        prob += (pulp.lpSum(ct1[i]) == marking_diff[i])
    prob.solve()
    dict1 = {'heurstic': pulp.value(prob.objective),
             'var': [pulp.value(var[i]) for i in range(trans_num)]}
    return dict1['heurstic'], np.array(dict1['var'])


def compute_exact_heuristic(h_score, solution_x, cost_vec, trans_index):
    result_aux = [x * 0 for x in solution_x]
    trust = False
    if solution_x[trans_index] >= 1:
        trust = True
    result_aux[trans_index] = 1
    result = solution_x - result_aux
    new_h_score = h_score - cost_vec[trans_index]
    return new_h_score, result, trust


def vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function):
    ini_vec = incidence_matrix.encode_marking(ini)
    fin_vec = incidence_matrix.encode_marking(fin)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[incidence_matrix.transitions[t]] = cost_function[t]
    return ini_vec, fin_vec, cost_vec


def construct_incident_matrix(net):
    p_index, t_index = {}, {}
    for p in net.places:
        p_index[p] = len(p_index)
    for t in net.transitions:
        t_index[t] = len(t_index)
    p_index_sort = sorted(p_index.items(), key=lambda kv: kv[0].name,reverse=True)
    t_index_sort = sorted(t_index.items(), key=lambda kv: kv[0].name,reverse=True)
    new_p_index = dict()
    for i in range(len(p_index_sort)):
        new_p_index[p_index_sort[i][0]] = i
    new_t_index = dict()
    for i in range(len(t_index_sort)):
        new_t_index[t_index_sort[i][0]] = i
    a_matrix = [[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))]
    for p in net.places:
        for a in p.in_arcs:
            a_matrix[new_p_index[p]][new_t_index[a.source]] += 1
        for a in p.out_arcs:
            a_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
    return a_matrix, new_p_index, new_t_index


def construct_consumption_matrix(net):
    p_index, t_index = {}, {}
    for p in net.places:
        p_index[p] = len(p_index)
    for t in net.transitions:
        t_index[t] = len(t_index)
    p_index_sort = sorted(p_index.items(), key=lambda kv: kv[0].name,reverse=True)
    t_index_sort = sorted(t_index.items(), key=lambda kv: kv[0].name,reverse=True)
    new_p_index = dict()
    for i in range(len(p_index_sort)):
        new_p_index[p_index_sort[i][0]] = i
    new_t_index = dict()
    for i in range(len(t_index_sort)):
        new_t_index[t_index_sort[i][0]] = i
    a_matrix = [[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))]
    for p in net.places:
        for a in p.out_arcs:
            a_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
    return a_matrix

def vectorize_initial_final_cost(place_index, trans_index, ini, fin, cost_function):
    ini_vec = encode_marking(ini, place_index)
    fini_vec = encode_marking(fin, place_index)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[trans_index[t]] = cost_function[t]
    return ini_vec, fini_vec, cost_vec


def encode_marking(marking, place_index):
    x = [0 for i in range(len(place_index))]
    for p in marking:
        x[place_index[p]] = marking[p]
    return x


class State:
    def __init__(self, f, g, h, t, m, m_tuple, p, solution_x):
        self.f = f
        self.g = g
        self.h = h
        self.t = t
        self.m = m
        self.mt = m_tuple
        self.p = p
        self.solution_x = solution_x
        # self.last_sync = last_sync

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif self.g < other.g:
            return True
        else:
            return self.h < other.h


def reconstruct_alignment(state, visited, queued, traversed, ret_tuple_as_trans_desc=False):
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
    result = {}
    result["alignment"] = alignment
    result["cost"] = state.g
    result["visited_states"] = visited
    result["queued"] =queued
    result["traversed"] = traversed
    return result



def construct_cost_function(sync_net):
    """
    Returns the standard cost function, which is:
    * event moves: cost 1
    * model moves: cost 1
    * tau moves: cost 0
    * sync moves: cost 0
    :param synchronous_product_net:
    :return:
    """
    costs = {}
    for t in sync_net.transitions:
        if t.label[0] == t.label[1] or (t.label[0] == '>>' and t.label[1] == chr(964)):
            costs[t] = 0
        else:
            costs[t] = 1
    return costs


def log_move(t):
    return t.label[0] != ">>" and t.label[1] == ">>"


def model_move(t):
    return t.label[0] == ">>" and t.label[1] != ">>"


def construct_model_net():
    model_net = PetriNet("model_net")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    p3 = PetriNet.Place("p3")
    p4 = PetriNet.Place("p4")
    p5 = PetriNet.Place("p5")
    p6 = PetriNet.Place("p6")
    p7 = PetriNet.Place("p7")
    model_net.places.add(p1)
    model_net.places.add(p2)
    model_net.places.add(p3)
    model_net.places.add(p4)
    model_net.places.add(p5)
    model_net.places.add(p6)
    model_net.places.add(p7)

    # add transitions for model
    t1 = PetriNet.Transition("t1", "a")
    t2 = PetriNet.Transition("t2", "b")
    t3 = PetriNet.Transition("t3", "c")
    t4 = PetriNet.Transition("t4", "d")
    t5 = PetriNet.Transition("t5", "e")
    t6 = PetriNet.Transition("t6", "e")
    t7 = PetriNet.Transition("t7", chr(964))
    t8 = PetriNet.Transition("t8", "g")
    t9 = PetriNet.Transition("t9", "h")
    model_net.transitions.add(t1)
    model_net.transitions.add(t2)
    model_net.transitions.add(t3)
    model_net.transitions.add(t4)
    model_net.transitions.add(t5)
    model_net.transitions.add(t6)
    model_net.transitions.add(t7)
    model_net.transitions.add(t8)
    model_net.transitions.add(t9)

    # add arcs for model
    petri_utils.add_arc_from_to(p1, t1, model_net)
    petri_utils.add_arc_from_to(t1, p2, model_net)
    petri_utils.add_arc_from_to(t1, p3, model_net)
    petri_utils.add_arc_from_to(p2, t2, model_net)
    petri_utils.add_arc_from_to(p2, t3, model_net)
    petri_utils.add_arc_from_to(p3, t4, model_net)
    petri_utils.add_arc_from_to(p3, t5, model_net)
    petri_utils.add_arc_from_to(t2, p4, model_net)
    petri_utils.add_arc_from_to(t3, p4, model_net)
    petri_utils.add_arc_from_to(p4, t5, model_net)
    petri_utils.add_arc_from_to(p4, t6, model_net)
    petri_utils.add_arc_from_to(t4, p5, model_net)
    petri_utils.add_arc_from_to(t5, p6, model_net)
    petri_utils.add_arc_from_to(t6, p6, model_net)
    petri_utils.add_arc_from_to(p5, t6, model_net)
    petri_utils.add_arc_from_to(p6, t7, model_net)
    petri_utils.add_arc_from_to(p6, t8, model_net)
    petri_utils.add_arc_from_to(p6, t9, model_net)
    petri_utils.add_arc_from_to(t7, p2, model_net)
    petri_utils.add_arc_from_to(t7, p3, model_net)
    petri_utils.add_arc_from_to(t8, p7, model_net)
    petri_utils.add_arc_from_to(t9, p7, model_net)

    # add marking for model
    model_im = Marking()
    model_im[p1] = 1
    model_fm = Marking()
    model_fm[p7] = 1
    # gviz = visualizer.apply(model_net, model_im, model_fm)
    # visualizer.view(gviz)
    return model_net, model_im, model_fm


def construct_trace_net():
    trace_net = PetriNet("trace_net")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    p3 = PetriNet.Place("p3")
    p4 = PetriNet.Place("p4")
    p5 = PetriNet.Place("p5")
    p6 = PetriNet.Place("p6")
    trace_net.places.add(p1)
    trace_net.places.add(p2)
    trace_net.places.add(p3)
    trace_net.places.add(p4)
    trace_net.places.add(p5)
    trace_net.places.add(p6)

    # add transitions for trace
    t1 = PetriNet.Transition("t1", "a")
    t2 = PetriNet.Transition("t2", "d")
    t3 = PetriNet.Transition("t3", "d")
    t4 = PetriNet.Transition("t4", "e")
    t5 = PetriNet.Transition("t5", "g")

    trace_net.transitions.add(t1)
    trace_net.transitions.add(t2)
    trace_net.transitions.add(t3)
    trace_net.transitions.add(t4)
    trace_net.transitions.add(t5)

    # add arcs for trace
    petri_utils.add_arc_from_to(p1, t1, trace_net)
    petri_utils.add_arc_from_to(p2, t2, trace_net)
    petri_utils.add_arc_from_to(p3, t3, trace_net)
    petri_utils.add_arc_from_to(p4, t4, trace_net)
    petri_utils.add_arc_from_to(p5, t5, trace_net)
    petri_utils.add_arc_from_to(t1, p2, trace_net)
    petri_utils.add_arc_from_to(t2, p3, trace_net)
    petri_utils.add_arc_from_to(t3, p4, trace_net)
    petri_utils.add_arc_from_to(t4, p5, trace_net)
    petri_utils.add_arc_from_to(t5, p6, trace_net)

    # add marking for model
    trace_im = Marking()
    trace_im[p1] = 1
    trace_fm = Marking()
    trace_fm[p6] = 1

    # gviz2 = visualizer.apply(trace_net, trace_im, trace_fm)
    # visualizer.view(gviz2)
    return trace_net, trace_im, trace_fm


def copy_into(source_net, target_net, upper, skip):
    t_map = {}
    p_map = {}
    for t in source_net.transitions:
        name = (t.name, skip) if upper else (skip, t.name)
        label = (t.label, skip) if upper else (skip, t.label)
        t_map[t] = petri.petrinet.PetriNet.Transition(name, label)
        target_net.transitions.add(t_map[t])

    for p in source_net.places:
        name = (p.name, skip) if upper else (skip, p.name)
        p_map[p] = petri.petrinet.PetriNet.Place(name)
        target_net.places.add(p_map[p])

    for t in source_net.transitions:
        for a in t.in_arcs:
            petri_utils.add_arc_from_to(p_map[a.source], t_map[t], target_net)
        for a in t.out_arcs:
            petri_utils.add_arc_from_to(t_map[t], p_map[a.target], target_net)

    return t_map, p_map