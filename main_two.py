ret_tuple_as_trans_desc = False

from pm4py.objects.petri import reachability_graph
from astar_implementation import utilities, thesis_two, trace_net
from pm4py.objects.petri import synchronous_product
from pm4py.objects.petri.utils import decorate_places_preset_trans, decorate_transitions_prepostset


if __name__ == '__main__':
    model_net, model_im, model_fm = utilities.construct_model_net()
    trace_net, trace_im, trace_fm = utilities.construct_trace_net_without_loop()
    sync_net, sync_im, sync_fm = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net, model_im, model_fm,'>>')
    cost_function = utilities.construct_cost_function(sync_net)
    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)
    consumption_matrix = utilities.construct_consumption_matrix(sync_net)
    incidence_matrix, p_index, t_index = utilities.construct_incident_matrix(sync_net)
    possible_enabling_transitions = set()
    for p in sync_im:
        for t in p.ass_trans:
            if t.label[0] == t.label[1]:
                a = t
    place_map = {}
    name = sorted(p_index, key=lambda k: k.name[1])
    for p in name:
        if p.name[0] == '>>':
            place_map[p] = (str(p.name[1]) + '\'')
        else:
            place_map[p] = (str(p.name[0]))
    split_lst = [None]
    print(p_index)
    print(t_index)
    print(incidence_matrix)
    ts = reachability_graph.construct_reachability_graph(sync_net, sync_im)
    sync_trans = []
    for k,v in t_index.items():
        if k.label[0] == k.label[1]:
            sync_trans.append(v)
    print(sync_trans)
    aux_dict = {'t_index': t_index, 'p_index': p_index, 'incidence_matrix': incidence_matrix,
                'consumption_matrix': consumption_matrix, 'place_map': place_map, 'cost_function': cost_function,
                'visited': 0, 'order': 0, 'transition_system': ts, 'sync_trans': sync_trans, 'state_to_check': [],
                'ts': ts}
    align = thesis_two.astar_with_split(sync_net, sync_im, sync_fm, aux_dict, split_lst)




