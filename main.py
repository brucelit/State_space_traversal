from pm4py.objects.petri import synchronous_product
from pm4py.objects.petri.utils import decorate_places_preset_trans, decorate_transitions_prepostset
from pm4py.objects.petri import reachability_graph
from astar_implementation import utilities, thesis_one
from pm4py.visualization.transition_system import visualizer as ts_visualizer

model_net, model_im, model_fm = utilities.construct_model_net()
trace_net, trace_im, trace_fm = utilities.construct_trace_net()
# sync_net, sync_im, sync_fm, cost_function = utilities.construct_sync_net(trace_net, trace_im, trace_fm, model_net, model_im, model_fm,
#                                                      ">>")
sync_net, sync_im, sync_fm = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net, model_im, model_fm,
                                                           ">>")
cost_function = utilities.construct_cost_function(sync_net)
# print(cost_function)
decorate_transitions_prepostset(sync_net)
decorate_places_preset_trans(sync_net)
# align = thesis_one.astar(sync_net, sync_im, sync_fm, cost_function)


ts = reachability_graph.construct_reachability_graph(sync_net, sync_im)
gviz = ts_visualizer.apply(ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
ts_visualizer.view(gviz)