ret_tuple_as_trans_desc = False

from pm4py.objects.petri import reachability_graph
from astar_implementation import utilities, thesis_two
from pm4py.objects.petri import synchronous_product
from pm4py.objects.petri.utils import decorate_places_preset_trans, decorate_transitions_prepostset


model_net, model_im, model_fm = utilities.construct_model_net()
trace_net, trace_im, trace_fm = utilities.construct_trace_net()
sync_net, sync_im, sync_fm = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net, model_im, model_fm,">>")
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
    if p.name[0] == ">>":
        place_map[p] = (str(p.name[1]) + "\'")
    else:
        place_map[p] = (str(p.name[0]))
split_lst = [None,a]
visited = 0
print(t_index)
ts = reachability_graph.construct_reachability_graph(sync_net, sync_im)
align = thesis_two.astar_with_split(1, 0, ts, place_map, sync_net, sync_im, sync_fm, consumption_matrix, incidence_matrix, p_index, t_index, cost_function, split_lst, visited)



