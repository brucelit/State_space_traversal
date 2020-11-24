from pm4py.objects.petri import synchronous_product
from pm4py.objects.petri.utils import decorate_places_preset_trans, decorate_transitions_prepostset
from pm4py.objects.petri import reachability_graph
from astar_implementation import utilities, thesis_one
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.visualization.petrinet import visualizer as viz
from astar_implementation import visualization as viz




model_net, model_im, model_fm = utilities.construct_model_net_without_loop()

trace_net, trace_im, trace_fm = utilities.construct_trace_net_without_loop()
# sync_net, sync_im, sync_fm, cost_function = utilities.construct_sync_net(trace_net, trace_im, trace_fm, model_net, model_im, model_fm,
#                                                      ">>")
sync_net, sync_im, sync_fm = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net, model_im, model_fm,
                                                           ">>")
incidence_matrix, p_index, t_index = utilities.construct_incident_matrix(sync_net)
place_map = {}
name = sorted(p_index, key=lambda k: k.name, reverse=True)
for p in name:
    if p.name[0] == ">>":
        place_map[p] = (str(p.name[1]) + "\'")
    else:
        place_map[p] = (str(p.name[0]) + "\'")
print(place_map)
# gviz = viz.apply(sync_net, sync_im, sync_fm )
# viz.view(gviz)
cost_function = utilities.construct_cost_function(sync_net)
print(cost_function)
decorate_transitions_prepostset(sync_net)
decorate_places_preset_trans(sync_net)
align = thesis_one.astar(sync_net, sync_im, sync_fm, cost_function)
# ts = reachability_graph.construct_reachability_graph(sync_net, sync_im)
# gviz,ts = viz.visualize_transition_system(ts)
# ts_visualizer.view(gviz)