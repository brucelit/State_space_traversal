import numpy as np

from pm4py.objects.petri import reachability_graph
from pm4py.objects.petri.utils import decorate_places_preset_trans, decorate_transitions_prepostset


def initialize_aux_dict(sync_net, sync_im, sync_fm, sync_index):
    incidence_matrix, consumption_matrix, p_index, t_index = construct_incident_consumption_matrix(sync_net)
    cost_function = construct_cost_function(sync_net)
    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)
    place_map = {}
    name = sorted(p_index, key=lambda place: place.name[1])
    for p in name:
        if p.name[0] == '>>':
            place_map[p] = (str(p.name[1]) + '\'')
        else:
            place_map[p] = (str(p.name[0]))
    ts = reachability_graph.construct_reachability_graph(sync_net, sync_im)
    sync_trans = []
    for k, v in t_index.items():
        if k.label[0] == k.label[1]:
            sync_trans.append(v)
    trace_trans = []
    for k, v in t_index.items():
        if k.label[1] == ">>":
            trace_trans.append(v)
    sync_map = {}
    for k1, v1 in t_index.items():
        for k2, v2 in sync_index.items():
            if k1 == k2:
                sync_map[v1] = v2
    ini_vec, fin_vec, cost_vec = vectorize_initial_final_cost(sync_im, sync_fm, p_index, t_index,
                                                              cost_function)
    x_0 = []
    trans_aux_dict = {}
    for k, v in t_index.items():
        if k.label[0] == k.label[1]:
            trans_aux_dict[v] = 0
        else:
            trans_aux_dict[v] = 1
    aux_dict = {'t_index': t_index, 'p_index': p_index, 'incidence_matrix': incidence_matrix,
                'consumption_matrix': consumption_matrix, 'place_map': place_map, 'cost_function': cost_function,
                'visited': 0, 'order': 0, 'transition_system': ts, 'sync_trans': sync_trans, 'state_to_check': [],
                'ts': ts, 'ini_vec': ini_vec, 'fin_vec': fin_vec, 'cost_vec': cost_vec,
                'traversed': 0, 'queued': 0, 'trace_trans': trace_trans, 'sync_index': sync_index,
                'trace_trans': trace_trans, 'x_0': x_0, 'sync_map': sync_map,"trans_aux_dict":trans_aux_dict,
                'block': 0, "recalculation": 0, 'split_lst':{None: -1}, 'reuse_flag': 0}
    return aux_dict


def vectorize_initial_final_cost(ini, fin, place_index, trans_index, cost_function):
    ini_vec = encode_marking(ini, place_index)
    fin_vec = encode_marking(fin, place_index)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[trans_index[t]] = cost_function[t]
    return ini_vec, fin_vec, cost_vec


def encode_marking(marking, place_index):
    x = [0 for i in range(len(place_index))]
    for p in marking:
        x[place_index[p]] = marking[p]
    return x


def construct_incident_consumption_matrix(sync_net):
    p_index, t_index = {}, {}
    for p in sync_net.places:
        p_index[p] = len(p_index)
    for t in sync_net.transitions:
        t_index[t] = len(t_index)
    p_index_sort = sorted(p_index.items(), key=lambda kv: kv[0].name, reverse=True)
    t_index_sort = sorted(t_index.items(), key=lambda kv: kv[0].name, reverse=True)
    new_p_index = dict()
    for i in range(len(p_index_sort)):
        new_p_index[p_index_sort[i][0]] = i
    new_t_index = dict()
    for i in range(len(t_index_sort)):
        new_t_index[t_index_sort[i][0]] = i
    incidence_matrix = [[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))]
    consumption_matrix = [[0 for i in range(len(new_t_index))] for j in range(len(new_p_index))]
    for p in sync_net.places:
        for a in p.in_arcs:
            incidence_matrix[new_p_index[p]][new_t_index[a.source]] += 1
        for a in p.out_arcs:
            incidence_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
            consumption_matrix[new_p_index[p]][new_t_index[a.target]] -= 1
    return incidence_matrix, consumption_matrix, new_p_index, new_t_index


def construct_cost_function(sync_net):
    costs = {}
    for t in sync_net.transitions:
        if t.label[0] == t.label[1] or (t.label[0] == '>>' and t.label[1] =='τ'):
            costs[t] = 0
        else:
            costs[t] = 1
    return costs
