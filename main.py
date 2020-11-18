from astar_implementation import utilities, thesis_one
from pm4py.objects.petri import synchronous_product
from pm4py.objects import petri
import numpy as np
from pm4py.visualization.petrinet.common import visualize
from pm4py.visualization.petrinet import visualizer
from astar_implementation import visualization
from pm4py.objects.petri.utils import decorate_places_preset_trans, decorate_transitions_prepostset


model_net, model_im, model_fm = utilities.construct_model_net()
trace_net, trace_im, trace_fm = utilities.construct_trace_net()
# sync_net, sync_im, sync_fm, cost_function = utilities.construct_sync_net(trace_net, trace_im, trace_fm, model_net, model_im, model_fm,
#                                                      ">>")
sync_net, sync_im, sync_fm = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net, model_im, model_fm,">>")
cost_function = utilities.construct_cost_function(sync_net)
# print(cost_function)
decorate_transitions_prepostset(sync_net)
decorate_places_preset_trans(sync_net)
align = thesis_one.astar(sync_net, sync_im, sync_fm, cost_function)

# incidence_matrix = petri.incidence_matrix.construct(sync_net)
# incidence_matrix = petri.incidence_matrix.construct(sync_net)
# a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
#
# enabled_trans = []
# for trans in incidence_matrix.transitions:
#     if trans.label[1] == ">>":
#         enabled_trans.append(trans)
# trans_index = []
# for trans in enabled_trans:
#     print(trans)
#     t_index = incidence_matrix.transitions[trans]
#     print(a_matrix[t_index])
# n = 10 # maximum number of b