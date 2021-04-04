from pm4py.objects.petri.importer.variants.pnml import import_net
from astar_implementation import construction, astar, rewrite_astar, initialization, synchronous_product
from csv import DictWriter
import time
import func_timeout
import pickle
from pm4py.objects.petri.importer.variants.pnml import import_net
from pm4py.algo.conformance.alignments.variants import state_equation_a_star, state_equation_less_memory
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri.importer.variants.pnml import import_net

field_names = ['alignment', 'cost', 'queued_states', 'visited_states', 'traversed_arcs', 'recalculation', 'time', 'lp_solved']
event_log = xes_importer.apply('F:\State_space_traversal\data\log_b.xes')
model_net, model_im, model_fm = import_net('F:\State_space_traversal\data\model_b.pnml')
for case_index, case in enumerate(event_log):

    trace_lst = []
    for event_index, event in enumerate(event_log[case_index]):
        trace_lst.append(event['concept:name'])

    trace_net, trace_im, trace_fm = construction.construct_trace(trace_lst)
    sync_net, sync_im, sync_fm, sync_index = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net,
                                                                           model_im,
                                                                           model_fm, '>>')
    aux_dict = initialization.initialize_aux_dict(sync_net, sync_im, sync_fm, sync_index)
    start_time = time.time()
    align = rewrite_astar.astar_with_split(sync_net, sync_im, sync_fm, aux_dict)
    align['time'] = time.time() - start_time
    print(align['cost'])
    # try:
    #      align = state_equation_a_star.apply(event_log[case_index], model_net, model_im, model_fm)
    #     align['time'] = time.time() - start_time
    #     print(align)
    #     with open('F:\State_space_traversal\cc_results\pm4py_2020b.csv', 'a') as f_object:
    #         dictwriter_object = DictWriter(f_object, fieldnames=field_names)
    #         # Pass the dictionary as an argument to the Writerow()
    #         dictwriter_object.writerow(align)
    #         # Close the file object
    #         f_object.close()
    # except func_timeout.exceptions.FunctionTimedOut:
    #     print("timeout", id)
    #     align = {'alignment': "??", 'cost': "??", 'visited_states': "??", 'queued_states': "??", 'traversed_arcs': "??",
    #             'recalculation': "??",'time':'??'}
    #     with open('F:\State_space_traversal\cc_results\pm4py_2020b.csv', 'a') as f_object:
    #         dictwriter_object = DictWriter(f_object, fieldnames=field_names)
    #         # Pass the dictionary as an argument to the Writerow()
    #         dictwriter_object.writerow(align)
    #         # Close the file object
    #         f_object.close()

# sync_net, sync_im, sync_fm, sync_index = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net,
#                                                                        model_im,
#                                                                        model_fm, '>>')
# aux_dict = initialization.initialize_aux_dict(sync_net, sync_im, sync_fm, sync_index)
# split_lst = {None: -1}
# start_time = time.time()
# # print(aux_dict['cost_vec'])
# # print(aux_dict['t_index'])
# print("", aux_dict['incidence_matrix'])
# align = astar_latest.astar_with_split(sync_net, sync_im, sync_fm, aux_dict, split_lst)
# print(align)
# print("--- %s seconds ---" % (time.time() - start_time))
# # field_names = ['trace', 'cost', 'visited_states', 'queued_states', 'traversed_arcs']
# with open('comparison2.csv', 'a') as f_object:
#     dictwriter_object = DictWriter(f_object, fieldnames=field_names)
#
#     # Pass the dictionary as an argument to the Writerow()
#     dictwriter_object.writerow(align)
#
#     # Close the file object
#     f_object.close()
